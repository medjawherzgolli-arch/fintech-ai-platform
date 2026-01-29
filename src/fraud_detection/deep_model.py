"""
Fraud Detection Deep Learning Model

PyTorch neural network for fraud detection with:
- MLP architecture (128 → 64 → 32 → 1)
- Weighted BCE loss for handling class imbalance
- Same interface as FraudDetectionModel
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


class FraudDetectionNetwork(nn.Module):
    """Neural network architecture for fraud detection."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3
    ):
        """
        Initialize the network.

        Args:
            input_size: Number of input features
            hidden_sizes: Tuple of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()

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

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class FraudDetectionDeepModel:
    """Fraud detection using PyTorch neural network."""

    def __init__(
        self,
        hidden_sizes: Tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 50,
        random_state: int = 42,
        device: Optional[str] = None
    ):
        """
        Initialize the deep learning fraud detection model.

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
        self.feature_names = None
        self.input_size = None

        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Convert timestamp to features if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour_sin'] = np.sin(2 * np.pi * df['transaction_hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['transaction_hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['transaction_day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['transaction_day'] / 7)

        # One-hot encode merchant category
        if 'merchant_category' in df.columns:
            df = pd.get_dummies(df, columns=['merchant_category'], prefix='merchant')

        return df

    def prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features and target.

        Args:
            df: Input DataFrame
            is_training: Whether this is for training (sets feature_names)

        Returns:
            Tuple of (features, target)
        """
        # Select base features
        base_features = [
            'transaction_amount', 'card_present', 'distance_from_home',
            'transaction_frequency_24h', 'avg_transaction_amount',
            'days_since_last_transaction', 'transactions_last_7days',
            'failed_transactions_24h', 'merchant_risk_score',
            'international', 'device_id_change'
        ]

        # Add derived features
        if 'hour_sin' in df.columns:
            base_features.extend(['hour_sin', 'hour_cos', 'day_sin', 'day_cos'])

        # Add merchant categories (exclude merchant_risk_score which is already in base)
        merchant_cols = [col for col in df.columns if col.startswith('merchant_') and col != 'merchant_risk_score']
        base_features.extend(merchant_cols)

        # Filter to existing columns
        feature_cols = [col for col in base_features if col in df.columns]

        if is_training:
            # During training, set the feature names
            self.feature_names = feature_cols
            self.input_size = len(feature_cols)
            X = df[feature_cols].copy()
        else:
            # During inference, use stored feature names and handle missing columns
            X = pd.DataFrame()
            for col in self.feature_names:
                if col in df.columns:
                    X[col] = df[col]
                else:
                    X[col] = 0  # Fill missing columns with 0

        y = df['is_fraud'] if 'is_fraud' in df.columns else None

        return X, y

    def _prepare_tensors(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        fit_scaler: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Convert DataFrame to tensors."""
        X_values = X.values
        if fit_scaler:
            X_values = self.scaler.fit_transform(X_values)
        else:
            X_values = self.scaler.transform(X_values)

        X_tensor = torch.FloatTensor(X_values).to(self.device)

        y_tensor = None
        if y is not None:
            y_tensor = torch.FloatTensor(y.values).unsqueeze(1).to(self.device)

        return X_tensor, y_tensor

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        early_stopping_patience: int = 10
    ) -> Dict[str, Any]:
        """
        Train the fraud detection model.

        Args:
            df: Training DataFrame
            test_size: Proportion of test set
            early_stopping_patience: Epochs to wait for improvement

        Returns:
            Dictionary with training metrics
        """
        # Preprocess
        df = self.preprocess(df)

        # Prepare features
        X, y = self.prepare_features(df, is_training=True)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Prepare tensors
        X_train_tensor, y_train_tensor = self._prepare_tensors(X_train, y_train, fit_scaler=True)
        X_test_tensor, y_test_tensor = self._prepare_tensors(X_test, y_test, fit_scaler=False)

        # Calculate class weights for imbalanced data
        n_positive = y_train.sum()
        n_negative = len(y_train) - n_positive
        pos_weight = torch.tensor([n_negative / n_positive]).to(self.device)

        # Initialize model
        self.model = FraudDetectionNetwork(
            input_size=self.input_size,
            hidden_sizes=self.hidden_sizes,
            dropout=self.dropout
        ).to(self.device)

        # Loss with class weighting and optimizer
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Create DataLoader (no weighted sampler - rely on pos_weight in loss only)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Modify model for BCE with logits (remove sigmoid from network)
        # We'll apply sigmoid manually for prediction
        self.model.network[-1] = nn.Identity()

        # Training loop
        best_loss = float('inf')
        patience_counter = 0
        train_losses = []

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(X_batch)
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
            logits = self.model(X_test_tensor).cpu().numpy().flatten()
            y_pred_proba = 1 / (1 + np.exp(-logits))  # Sigmoid
            y_pred = (y_pred_proba >= 0.5).astype(int)

        y_test_np = y_test.values

        metrics = {
            'accuracy': accuracy_score(y_test_np, y_pred),
            'precision': precision_score(y_test_np, y_pred, zero_division=0),
            'recall': recall_score(y_test_np, y_pred, zero_division=0),
            'f1_score': f1_score(y_test_np, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test_np, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test_np, y_pred).tolist(),
            'classification_report': classification_report(y_test_np, y_pred, zero_division=0),
            'epochs_trained': epoch + 1,
            'final_train_loss': train_losses[-1],
            'class_imbalance_ratio': f"1:{int(n_negative/n_positive)}"
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Detect fraud in transactions.

        Args:
            df: Input DataFrame

        Returns:
            Array of fraud predictions (1 = fraud, 0 = legitimate)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        proba = self.predict_proba(df)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fraud probability.

        Args:
            df: Input DataFrame

        Returns:
            Array of fraud probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess
        df = self.preprocess(df)

        # Prepare features
        X, _ = self.prepare_features(df)

        # Prepare tensors
        X_tensor, _ = self._prepare_tensors(X)

        # Predict
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor).cpu().numpy().flatten()
            proba = 1 / (1 + np.exp(-logits))  # Sigmoid

        return proba

    def save(self, filepath: str):
        """Save the model."""
        model_state = self.model.state_dict() if self.model else None

        joblib.dump({
            'model_state': model_state,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'dropout': self.dropout
        }, filepath)

    def load(self, filepath: str):
        """Load the model."""
        data = joblib.load(filepath)

        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.input_size = data['input_size']
        self.hidden_sizes = data['hidden_sizes']
        self.dropout = data['dropout']

        # Rebuild and load model
        if data['model_state'] is not None:
            self.model = FraudDetectionNetwork(
                input_size=self.input_size,
                hidden_sizes=self.hidden_sizes,
                dropout=self.dropout
            ).to(self.device)
            # Remove sigmoid for BCEWithLogitsLoss compatibility
            self.model.network[-1] = nn.Identity()
            self.model.load_state_dict(data['model_state'])
            self.model.eval()


def main():
    """Example usage of the deep fraud detection model."""
    from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator

    # Generate data
    print("Generating synthetic data...")
    generator = FraudDetectionDataGenerator(random_state=42)
    df = generator.generate(n_samples=50000, fraud_rate=0.02)

    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")

    # Train model
    print("\nTraining deep learning fraud detection model...")
    model = FraudDetectionDeepModel(
        hidden_sizes=(128, 64, 32),
        dropout=0.3,
        epochs=50,
        batch_size=256,
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
    model.save("models/fraud_detection/fraud_detection_deep_model.pkl")
    print("\nModel saved to models/fraud_detection/fraud_detection_deep_model.pkl")


if __name__ == "__main__":
    main()
