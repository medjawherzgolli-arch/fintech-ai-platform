"""
Credit Risk Model Pipeline

Machine learning pipeline for credit risk assessment including:
- Data preprocessing
- Feature engineering
- Model training (XGBoost, LightGBM, CatBoost)
- Model evaluation
- Prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


class CreditRiskModel:
    """Credit risk prediction model."""

    def __init__(self, model_type: str = "xgboost", random_state: int = 42):
        """
        Initialize the credit risk model.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', 'catboost')
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = ['home_ownership', 'loan_purpose']

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
                else:
                    # Handle unseen categories
                    df[col] = df[col].apply(
                        lambda x: x if x in self.label_encoders[col].classes_ else self.label_encoders[col].classes_[0]
                    )
                    df[col] = self.label_encoders[col].transform(df[col])

        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
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

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train the credit risk model.

        Args:
            df: Training DataFrame
            test_size: Proportion of test set
            cv_folds: Number of cross-validation folds

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

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model based on type
        if self.model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='logloss'
            )
        elif self.model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_type == "catboost":
            self.model = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                verbose=False
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Train
        self.model.fit(X_train_scaled, y_train)

        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train,
            cv=cv_folds, scoring='roc_auc'
        )

        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        # Evaluate
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }

        return metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            df: Input DataFrame

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Preprocess
        df = self.preprocess(df, is_training=False)

        # Prepare features
        X, _ = self.prepare_features(df)

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions

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

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict probabilities
        proba = self.model.predict_proba(X_scaled)[:, 1]

        return proba

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance.

        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.model_type in ["xgboost", "lightgbm"]:
            importance = self.model.feature_importances_
        elif self.model_type == "catboost":
            importance = self.model.get_feature_importance()
        else:
            raise ValueError(f"Feature importance not supported for {self.model_type}")

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
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }, filepath)

    def load(self, filepath: str):
        """Load the model."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.label_encoders = data['label_encoders']
        self.feature_names = data['feature_names']
        self.model_type = data['model_type']


def main():
    """Example usage of the credit risk model."""
    from src.data_generators.credit_risk_generator import CreditRiskDataGenerator

    # Generate data
    print("Generating synthetic data...")
    generator = CreditRiskDataGenerator(random_state=42)
    df = generator.generate(n_samples=10000, default_rate=0.15)

    # Train model
    print("\nTraining credit risk model...")
    model = CreditRiskModel(model_type="xgboost", random_state=42)
    metrics = model.train(df, test_size=0.2, cv_folds=5)

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

    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance())

    # Save model
    model.save("models/credit_risk/credit_risk_model.pkl")
    print("\nModel saved to models/credit_risk/credit_risk_model.pkl")


if __name__ == "__main__":
    main()
