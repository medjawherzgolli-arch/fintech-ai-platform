"""
Credit Risk Deep Learning Model

PyTorch neural network for credit risk assessment with:
- Embedding layers for categorical features
- MLP architecture with BatchNorm and Dropout
- Same interface as CreditRiskModel
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class CreditRiskNetwork(nn.Module):
    """Neural network architecture for credit risk prediction."""

    def __init__(
        self,
        n_continuous: int,
        embedding_dims: Dict[str, Tuple[int, int]],
        hidden_sizes: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.3
    ):
        """
        Initialize the network.

        Args:
            n_continuous: Number of continuous features
            embedding_dims: Dict mapping categorical feature names to (n_categories, embedding_dim)
            hidden_sizes: Tuple of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(num_embeddings=dims[0], embedding_dim=dims[1])
            for name, dims in embedding_dims.items()
        })

        # Calculate total input size
        embedding_total = sum(dims[1] for dims in embedding_dims.values())
        input_size = n_continuous + embedding_total

        # Build MLP layers
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_continuous: torch.Tensor, x_categorical: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass."""
        # Get embeddings for categorical features
        embedded = [self.embeddings[name](x_categorical[name]) for name in self.embeddings]

        # Concatenate continuous and embedded features
        if embedded:
            x = torch.cat([x_continuous] + embedded, dim=1)
        else:
            x = x_continuous

        return self.mlp(x)


class CreditRiskDeepModel:
    """Credit risk prediction using PyTorch neural network."""

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize the deep learning credit risk model.

        Args:
            hidden_sizes: Tuple of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Training batch size
            epochs: Number of training epochs
            random_state: Random seed
            device: Device to use ('cuda' or 'cpu')
        """
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state

        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = ['home_ownership', 'loan_purpose']
        self.continuous_features = None
        self.embedding_dims = {}

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def preprocess(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Preprocess the data.

        Args:
            df: Input DataFrame
            is_training: Whether this is training data

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Handle categorical features
        for col in self.categorical_features:
            if col in df.columns:
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col])
                    # Store number of categories for embedding
                    self.embedding_dims[col] = (
                        len(self.label_encoders[col].classes_),
                        min(50, (len(self.label_encoders[col].classes_) + 1) // 2)
                    )
                else:
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (features, target)
        """
        # Select features
        feature_cols = [
            'age', 'employment_length', 'annual_income', 'credit_score',
            'loan_amount', 'loan_term', 'interest_rate', 'debt_to_income',
            'home_ownership', 'loan_purpose', 'delinquencies_2yrs',
            'open_accounts', 'total_credit_lines', 'revolving_balance',
            'revolving_utilization'
        ]

        X = df[feature_cols].copy()
        y = df['default'] if 'default' in df.columns else None

        self.feature_names = feature_cols
        self.continuous_features = [f for f in feature_cols if f not in self.categorical_features]

        return X, y

    def _prepare_tensors(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """Convert DataFrame to tensors."""
        # Continuous features
        X_continuous = X[self.continuous_features].values
        if fit_scaler:
            X_continuous = self.scaler.fit_transform(X_continuous)
        else:
            X_continuous = self.scaler.transform(X_continuous)

        X_continuous_tensor = torch.FloatTensor(X_continuous).to(self.device)

        # Categorical features
        X_categorical = {
            col: torch.LongTensor(X[col].values).to(self.device)
            for col in self.categorical_features
        }

        # Target
        y_tensor = None
        if y is not None:
            y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(self.device)

        return X_continuous_tensor, X_categorical, y_tensor

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the credit risk model.

        Args:
            df: Training DataFrame
            test_size: Proportion of test set
            early_stopping_patience: Epochs to wait for improvement

        Returns:
            Dictionary with training metrics
        """
        # Preprocess
        df = self.preprocess(df, is_training=True)

        # Prepare features
        X, y = self.prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Prepare tensors
        X_train_cont, X_train_cat, y_train_tensor = self._prepare_tensors(
            X_train, y_train, fit_scaler=True
        )
        X_test_cont, X_test_cat, y_test_tensor = self._prepare_tensors(
            X_test, y_test, fit_scaler=False
        )

        # Initialize model
        self.model = CreditRiskNetwork(
            n_continuous=len(self.continuous_features),
            embedding_dims=self.embedding_dims,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create DataLoader for training
        train_dataset = TensorDataset(
            X_train_cont,
            *[X_train_cat[col] for col in self.categorical_features],
            y_train_tensor
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                X_cont_batch = batch[0]
                X_cat_batch = {
                    col: batch[i + 1]
                    for i, col in enumerate(self.categorical_features)
                }
                y_batch = batch[-1]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_cont_batch, X_cat_batch)
                loss = criterion(outputs, y_batch)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)

            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        # Evaluation
        self.model.eval()
        with torch.no_grad():
            y_pred_proba = self.model(X_test_cont, X_test_cat).cpu().numpy().flatten()
            y_pred = (y_pred_proba >= 0.5).astype(int)

        y_test_np = y_test.values

        metrics = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'precision': precision_score(y_test_np, y_pred),
            'recall': recall_score(y_test_np, y_pred),
            'f1_score': f1_score(y_test_np, y_pred),
            'roc_auc': roc_auc_score(y_test_np, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test_np, y_pred).tolist(),
            'classification_report': classification_report(y_test_np, y_pred),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_losses[-1]
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input DataFrame

        Returns:
            Array of predictions (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.predict_proba(df)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of default.

        Args:
            df: Input DataFrame

        Returns:
            Array of default probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess
        df = self.preprocess(df, is_training=False)

        # Prepare features
        X, _ = self.prepare_features(df)

        # Prepare tensors
        X_cont, X_cat, _ = self._prepare_tensors(X)

        # Predict
        self.model.eval()
        with torch.no_grad():
            proba = self.model(X_cont, X_cat).cpu().numpy().flatten()

        return proba

    def save(self, filepath: str):
        """Save the model."""
        # Move model to CPU for saving
        model_state = self.model.state_dict() if self.model else None

        joblib.dump({
            'model_state': model_state,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'continuous_features': self.continuous_features,
            'categorical_features': self.categorical_features,
            'embedding_dims': self.embedding_dims,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout
        }, filepath)

    def load(self, filepath: str):
        """Load the model."""
        data = joblib.load(filepath)

        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.continuous_features = data['continuous_features']
        self.categorical_features = data['categorical_features']
        self.embedding_dims = data['embedding_dims']
        self.hidden_sizes = data['hidden_sizes']
        self.dropout = data['dropout']

        # Rebuild and load model
        if data['model_state'] is not None:
            self.model = CreditRiskNetwork(
                n_continuous=len(self.continuous_features),
                embedding_dims=self.embedding_dims,
                hidden_sizes=self.hidden_sizes,
                dropout=self.dropout
            ).to(self.device)
            self.model.load_state_dict(data['model_state'])
            self.model.eval()


def main():
    """Example usage of the deep credit risk model."""
    from src.data_generators.credit_risk_generator import CreditRiskDataGenerator

    # Generate data
    print("Generating synthetic data...")
    generator = CreditRiskDataGenerator(random_state=42)
    df = generator.generate(n_samples=10000, default_rate=0.15)

    # Train model
    print("\nTraining deep learning credit risk model...")
    model = CreditRiskDeepModel(
        hidden_sizes=(64, 32, 16),
        dropout=0.3,
        epochs=50,
        random_state=42
    )
    metrics = model.train(df, test_size=0.2)

    # Display metrics
    print("\nModel Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Save model
    model.save("models/credit_risk/credit_risk_deep_model.pkl")
    print("\nModel saved to models/credit_risk/credit_risk_deep_model.pkl")


if __name__ == "__main__":
    main()
