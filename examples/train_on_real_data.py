"""
Example: Training Models on Real Data

This script demonstrates how to use real public datasets with the platform.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.credit_risk.model import CreditRiskModel
from src.fraud_detection.model import FraudDetectionModel
from src.utils.logger import setup_logger

import warnings
warnings.filterwarnings('ignore')


def train_german_credit():
    """Train on German Credit dataset."""
    logger = setup_logger("real_data_training", level="INFO")

    logger.info("=" * 70)
    logger.info("Training on German Credit Data (UCI)")
    logger.info("=" * 70)

    # Check if data exists
    data_path = Path("data/raw/credit_risk/german_credit.csv")

    if not data_path.exists():
        logger.error(f"Data not found at {data_path}")
        logger.info("Please run: python scripts/download_real_data.py")
        return

    # Load data
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Features: {df.shape[1]}")
    logger.info(f"Default rate: {df['default'].mean():.2%}")

    logger.info("\nData preview:")
    logger.info(df.head().to_string())

    # Prepare data for model
    # The German Credit dataset has categorical features that need encoding
    # For simplicity, we'll select numeric features

    # Extract numeric features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_cols.remove('default')  # Remove target

    logger.info(f"\nUsing {len(numeric_cols)} numeric features")

    # Create a dataframe with required format
    model_df = df[numeric_cols + ['default']].copy()

    # Train model
    logger.info("\nTraining XGBoost model...")
    model = CreditRiskModel(model_type="xgboost", random_state=42)

    try:
        # Note: The model expects specific feature names
        # For real use, you'd need to map German Credit features to expected features
        # or modify the model to accept arbitrary features

        logger.info("\nNote: German Credit dataset has different features than expected.")
        logger.info("For production use, implement feature mapping or use the original model structure.")

        # For demonstration, let's show the data structure
        logger.info("\nDataset structure:")
        logger.info(model_df.info())

    except Exception as e:
        logger.error(f"Error training model: {e}")
        logger.info("\nTo use this dataset, you need to:")
        logger.info("1. Map German Credit features to expected features")
        logger.info("2. Or modify CreditRiskModel to accept custom features")


def train_credit_card_fraud():
    """Train on Credit Card Fraud dataset (if available)."""
    logger = setup_logger("fraud_training", level="INFO")

    logger.info("\n" + "=" * 70)
    logger.info("Training on Credit Card Fraud Data (Kaggle)")
    logger.info("=" * 70)

    # Check if data exists
    data_path = Path("data/raw/fraud_detection/creditcard.csv")

    if not data_path.exists():
        logger.info(f"\nData not found at {data_path}")
        logger.info("To download:")
        logger.info("1. Setup Kaggle API: https://www.kaggle.com/docs/api")
        logger.info("2. Run: kaggle datasets download -d mlg-ulb/creditcardfraud")
        logger.info("3. Extract to: data/raw/fraud_detection/")
        return

    # Load data
    logger.info(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} records")
    logger.info(f"Features: {df.shape[1]}")
    logger.info(f"Fraud rate: {df['Class'].mean():.2%}")

    # Prepare data
    # Credit Card Fraud dataset has 'Class' as target (0=legitimate, 1=fraud)
    # Features are already anonymized and scaled (V1-V28)

    logger.info("\nThis dataset is ready to use!")
    logger.info("Features V1-V28 are PCA transformations")
    logger.info("Time: seconds elapsed from first transaction")
    logger.info("Amount: transaction amount")

    # Rename target column to match our model
    df_prepared = df.copy()
    df_prepared['is_fraud'] = df_prepared['Class']
    df_prepared = df_prepared.drop('Class', axis=1)

    # Add required features (dummy values if not present)
    required_features = [
        'transaction_amount', 'card_present', 'distance_from_home',
        'transaction_frequency_24h', 'avg_transaction_amount',
        'days_since_last_transaction', 'transactions_last_7days',
        'failed_transactions_24h', 'merchant_risk_score',
        'international', 'device_id_change'
    ]

    # Map existing features
    if 'Amount' in df_prepared.columns:
        df_prepared['transaction_amount'] = df_prepared['Amount']

    logger.info("\nTo use this dataset with FraudDetectionModel:")
    logger.info("1. Modify the model to accept V1-V28 features")
    logger.info("2. Or create feature mapping from V1-V28 to expected features")
    logger.info("3. Dataset is ready for training with minimal preprocessing")


def create_adapter_example():
    """Show how to create a data adapter for real datasets."""
    logger = setup_logger("adapter_example", level="INFO")

    logger.info("\n" + "=" * 70)
    logger.info("Creating Data Adapter for Real Datasets")
    logger.info("=" * 70)

    adapter_code = '''
# Example: Data Adapter for German Credit Dataset

import pandas as pd

class GermanCreditAdapter:
    """Adapter to convert German Credit data to platform format."""

    def __init__(self):
        self.feature_mapping = {
            'duration_months': 'loan_term',
            'credit_amount': 'loan_amount',
            'age': 'age',
            'installment_rate': 'debt_to_income',
            # Add more mappings as needed
        }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform German Credit format to platform format."""
        result = pd.DataFrame()

        # Map existing features
        for old_name, new_name in self.feature_mapping.items():
            if old_name in df.columns:
                result[new_name] = df[old_name]

        # Create derived features
        result['employment_length'] = 5  # Default value
        result['annual_income'] = df['credit_amount'] * 0.3  # Estimate
        result['credit_score'] = 700 - (df['credit_amount'] / 100)  # Estimate

        # Target variable
        result['default'] = df['default']

        return result

# Usage
adapter = GermanCreditAdapter()
german_df = pd.read_csv("data/raw/credit_risk/german_credit.csv")
platform_df = adapter.transform(german_df)

# Now train with platform
from src.credit_risk.model import CreditRiskModel
model = CreditRiskModel()
metrics = model.train(platform_df)
    '''

    logger.info("\nData Adapter Pattern:")
    logger.info(adapter_code)

    logger.info("\nThis pattern allows you to:")
    logger.info("1. Use any real dataset with the platform")
    logger.info("2. Map features to expected format")
    logger.info("3. Create derived features as needed")
    logger.info("4. Maintain separation between data sources and models")


def main():
    """Main function."""
    logger = setup_logger("main", level="INFO")

    logger.info("=" * 70)
    logger.info("FINTECH AI PLATFORM - REAL DATA TRAINING EXAMPLES")
    logger.info("=" * 70)

    # Try German Credit
    train_german_credit()

    # Try Credit Card Fraud
    train_credit_card_fraud()

    # Show adapter pattern
    create_adapter_example()

    logger.info("\n" + "=" * 70)
    logger.info("NEXT STEPS")
    logger.info("=" * 70)
    logger.info("\n1. Download datasets using: python scripts/download_real_data.py")
    logger.info("2. Create data adapters for your datasets")
    logger.info("3. Or modify models to accept custom feature sets")
    logger.info("4. See DATA_SOURCES.md for available datasets")


if __name__ == "__main__":
    main()
