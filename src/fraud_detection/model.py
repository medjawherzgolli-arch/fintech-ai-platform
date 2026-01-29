"""
Fraud Detection Model Pipeline

Machine learning pipeline for fraud detection including:
- Data preprocessing
- Feature engineering
- Anomaly detection models (Isolation Forest, Autoencoder, XGBoost)
- Model evaluation
- Real-time prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import xgboost as xgb


class FraudDetectionModel:
    """Fraud detection model using multiple approaches."""

    def __init__(self, model_type: str = "isolation_forest", random_state: int = 42):
        """
        Initialize the fraud detection model.

        Args:
            model_type: Type of model ('isolation_forest', 'xgboost', 'autoencoder')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

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

    def prepare_features(self, df: pd.DataFrame, is_training: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
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

        # Add merchant categories
        merchant_cols = [col for col in df.columns if col.startswith('merchant_')]
        base_features.extend(merchant_cols)

        # Filter to existing columns
        feature_cols = [col for col in base_features if col in df.columns]

        if is_training:
            # During training, set the feature names
            self.feature_names = feature_cols
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

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        contamination: float = 0.02
    ) -> Dict[str, Any]:
        """
        Train the fraud detection model.

        Args:
            df: Training DataFrame
            test_size: Proportion of test set
            contamination: Expected fraud rate for anomaly detection

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

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model based on type
        if self.model_type == "isolation_forest":
            self.model = IsolationForest(
                contamination=contamination,
                random_state=self.random_state,
                n_estimators=100
            )
            # Isolation Forest uses -1 for outliers, 1 for inliers
            self.model.fit(X_train_scaled)
            y_pred_train = (self.model.predict(X_train_scaled) == -1).astype(int)
            y_pred = (self.model.predict(X_test_scaled) == -1).astype(int)
            # Get anomaly scores (lower = more anomalous)
            scores = -self.model.score_samples(X_test_scaled)
            # Normalize scores to [0, 1]
            y_pred_proba = (scores - scores.min()) / (scores.max() - scores.min())

        elif self.model_type == "xgboost":
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
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

        # Preprocess
        df = self.preprocess(df)

        # Prepare features
        X, _ = self.prepare_features(df)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        if self.model_type == "isolation_forest":
            predictions = (self.model.predict(X_scaled) == -1).astype(int)
        else:
            predictions = self.model.predict(X_scaled)

        return predictions

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

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        if self.model_type == "isolation_forest":
            scores = -self.model.score_samples(X_scaled)
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        else:
            proba = self.model.predict_proba(X_scaled)[:, 1]

        return proba

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (only for XGBoost).

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.model_type != "xgboost":
            raise ValueError("Feature importance only available for XGBoost model")

        importance = self.model.feature_importances_

        df_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return df_importance

    def save(self, filepath: str):
        """Save the model."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)

    def load(self, filepath: str):
        """Load the model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']


def main():
    """Example usage of the fraud detection model."""
    from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator

    # Generate data
    print("Generating synthetic data...")
    generator = FraudDetectionDataGenerator(random_state=42)
    df = generator.generate(n_samples=50000, fraud_rate=0.02)

    # Train model
    print("\nTraining fraud detection model...")
    model = FraudDetectionModel(model_type="xgboost", random_state=42)
    metrics = model.train(df, test_size=0.2, contamination=0.02)

    # Display metrics
    print("\nModel Performance Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        if metric not in ['confusion_matrix', 'classification_report']:
            print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])

    print("\nClassification Report:")
    print(metrics['classification_report'])

    # Feature importance (if XGBoost)
    if model.model_type == "xgboost":
        print("\nTop 10 Features:")
        print(model.get_feature_importance().head(10))

    # Save model
    model.save("models/fraud_detection/fraud_detection_model.pkl")
    print("\nModel saved to models/fraud_detection/fraud_detection_model.pkl")


if __name__ == "__main__":
    main()
