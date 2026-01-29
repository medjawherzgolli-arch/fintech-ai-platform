"""
Download and Use Real Financial Data

This script downloads real public datasets and trains models on them.
"""

import sys
from pathlib import Path
import pandas as pd
import requests
from io import StringIO

sys.path.insert(0, str(Path(__file__).parent))

from src.credit_risk.model import CreditRiskModel
from src.fraud_detection.model import FraudDetectionModel
from src.utils.logger import setup_logger

logger = setup_logger("real_data", level="INFO")


def download_german_credit():
    """Download German Credit dataset from UCI."""
    logger.info("=" * 70)
    logger.info("DOWNLOADING GERMAN CREDIT DATA (UCI)")
    logger.info("=" * 70)

    data_dir = Path("data/raw/credit_risk")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download the data file
        logger.info("\nDownloading from UCI ML Repository...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

        response = requests.get(url)
        response.raise_for_status()

        # Column names for German Credit dataset
        columns = [
            'status_checking', 'duration_months', 'credit_history', 'purpose',
            'credit_amount', 'savings', 'employment_since', 'installment_rate',
            'personal_status', 'other_debtors', 'residence_since', 'property',
            'age', 'other_installments', 'housing', 'existing_credits',
            'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk'
        ]

        # Parse the data
        df = pd.read_csv(StringIO(response.text), sep=' ', names=columns, header=None)

        # Convert credit_risk: 1=Good, 2=Bad -> 0=No Default, 1=Default
        df['default'] = (df['credit_risk'] == 2).astype(int)
        df = df.drop('credit_risk', axis=1)

        # Save raw data
        output_file = data_dir / "german_credit_raw.csv"
        df.to_csv(output_file, index=False)

        logger.info(f"✅ Downloaded successfully!")
        logger.info(f"   Location: {output_file}")
        logger.info(f"   Records: {len(df)}")
        logger.info(f"   Default rate: {df['default'].mean():.2%}")

        return df

    except Exception as e:
        logger.error(f"❌ Error downloading data: {e}")
        return None


def prepare_german_credit_for_model(df):
    """Convert German Credit data to model format."""
    logger.info("\n" + "=" * 70)
    logger.info("PREPARING DATA FOR MODEL")
    logger.info("=" * 70)

    # Create a simplified dataset with numeric features
    prepared_df = pd.DataFrame()

    # Map features to expected format
    prepared_df['age'] = df['age']
    prepared_df['employment_length'] = df['employment_since'].map({
        'A71': 0, 'A72': 1, 'A73': 4, 'A74': 7, 'A75': 10
    }).fillna(5)

    # Estimate income from credit amount (rough approximation)
    prepared_df['annual_income'] = df['credit_amount'] * 0.4

    # Map credit scores (approximation from status_checking)
    credit_score_map = {
        'A11': 750, 'A12': 700, 'A13': 650, 'A14': 600
    }
    prepared_df['credit_score'] = df['status_checking'].map(credit_score_map).fillna(650)

    # Direct mappings
    prepared_df['loan_amount'] = df['credit_amount']
    prepared_df['loan_term'] = df['duration_months']

    # Estimate interest rate based on credit history
    interest_map = {
        'A30': 0.15, 'A31': 0.12, 'A32': 0.10, 'A33': 0.08, 'A34': 0.12
    }
    prepared_df['interest_rate'] = df['credit_history'].map(interest_map).fillna(0.12)

    # Calculate debt-to-income
    prepared_df['debt_to_income'] = (df['credit_amount'] / 12) / (prepared_df['annual_income'] / 12)
    prepared_df['debt_to_income'] = prepared_df['debt_to_income'].clip(0, 1)

    # Map home ownership
    home_map = {
        'A151': 'RENT', 'A152': 'OWN', 'A153': 'OTHER'
    }
    prepared_df['home_ownership'] = df['housing'].map(home_map).fillna('RENT')

    # Map loan purpose
    purpose_map = {
        'A40': 'credit_card', 'A41': 'credit_card', 'A42': 'home_improvement',
        'A43': 'other', 'A44': 'credit_card', 'A45': 'major_purchase',
        'A46': 'home_improvement', 'A47': 'other', 'A48': 'other',
        'A49': 'small_business', 'A410': 'other'
    }
    prepared_df['loan_purpose'] = df['purpose'].map(purpose_map).fillna('other')

    # Additional required features
    prepared_df['delinquencies_2yrs'] = 0  # Not in German dataset
    prepared_df['open_accounts'] = df['existing_credits']
    prepared_df['total_credit_lines'] = df['existing_credits'] + 5
    prepared_df['revolving_balance'] = df['credit_amount'] * 0.3
    prepared_df['revolving_utilization'] = 0.35  # Average assumption

    # Target variable
    prepared_df['default'] = df['default']

    logger.info(f"✅ Data prepared for model")
    logger.info(f"   Shape: {prepared_df.shape}")
    logger.info(f"   Features: {prepared_df.shape[1] - 1}")

    return prepared_df


def train_on_real_data(df):
    """Train model on real data."""
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING MODEL ON REAL DATA")
    logger.info("=" * 70)

    try:
        # Initialize model
        model = CreditRiskModel(model_type="xgboost", random_state=42)

        # Train
        logger.info("\nTraining XGBoost model...")
        metrics = model.train(df, test_size=0.2, cv_folds=5)

        logger.info("\n✅ MODEL TRAINED ON REAL DATA!")
        logger.info("\nPerformance Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC:   {metrics['roc_auc']:.4f}")

        # Feature importance
        logger.info("\nTop 5 Important Features:")
        importance = model.get_feature_importance()
        for idx, row in importance.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Save model
        model.save("models/credit_risk/credit_risk_model_real_data.pkl")
        logger.info("\n💾 Model saved to: models/credit_risk/credit_risk_model_real_data.pkl")

        return model, metrics

    except Exception as e:
        logger.error(f"❌ Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_real_data_model():
    """Test the model trained on real data."""
    logger.info("\n" + "=" * 70)
    logger.info("TESTING MODEL TRAINED ON REAL DATA")
    logger.info("=" * 70)

    try:
        # Load model
        model = CreditRiskModel()
        model.load("models/credit_risk/credit_risk_model_real_data.pkl")

        # Test case 1: Good applicant
        logger.info("\nTest 1: Good Credit Profile")
        good_applicant = pd.DataFrame([{
            'age': 35, 'employment_length': 8.0, 'annual_income': 85000.00,
            'credit_score': 750, 'loan_amount': 20000.00, 'loan_term': 36,
            'interest_rate': 0.08, 'debt_to_income': 0.20,
            'home_ownership': 'OWN', 'loan_purpose': 'home_improvement',
            'delinquencies_2yrs': 0, 'open_accounts': 3, 'total_credit_lines': 8,
            'revolving_balance': 5000.00, 'revolving_utilization': 0.25
        }])

        risk_prob = model.predict_proba(good_applicant)[0]
        decision = model.predict(good_applicant)[0]

        logger.info(f"  Default Risk: {risk_prob:.1%}")
        logger.info(f"  Decision: {'REJECT' if decision else 'APPROVE'}")

        # Test case 2: Risky applicant
        logger.info("\nTest 2: High Risk Profile")
        risky_applicant = pd.DataFrame([{
            'age': 23, 'employment_length': 1.0, 'annual_income': 32000.00,
            'credit_score': 600, 'loan_amount': 25000.00, 'loan_term': 60,
            'interest_rate': 0.15, 'debt_to_income': 0.50,
            'home_ownership': 'RENT', 'loan_purpose': 'other',
            'delinquencies_2yrs': 0, 'open_accounts': 2, 'total_credit_lines': 5,
            'revolving_balance': 8000.00, 'revolving_utilization': 0.80
        }])

        risk_prob = model.predict_proba(risky_applicant)[0]
        decision = model.predict(risky_applicant)[0]

        logger.info(f"  Default Risk: {risk_prob:.1%}")
        logger.info(f"  Decision: {'REJECT' if decision else 'APPROVE'}")

        logger.info("\n✅ Tests completed successfully!")

    except Exception as e:
        logger.error(f"❌ Error testing model: {e}")


def main():
    """Main function."""
    logger.info("=" * 70)
    logger.info("GET REAL DATA & TRAIN MODELS")
    logger.info("=" * 70)

    # Step 1: Download data
    df_raw = download_german_credit()

    if df_raw is None:
        logger.error("Failed to download data. Exiting.")
        return

    # Step 2: Prepare data
    df_prepared = prepare_german_credit_for_model(df_raw)

    # Save prepared data
    output_file = Path("data/processed/german_credit_prepared.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_prepared.to_csv(output_file, index=False)
    logger.info(f"\n💾 Prepared data saved to: {output_file}")

    # Step 3: Train model
    model, metrics = train_on_real_data(df_prepared)

    if model is None:
        logger.error("Failed to train model. Exiting.")
        return

    # Step 4: Test model
    test_real_data_model()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ SUCCESS!")
    logger.info("=" * 70)
    logger.info("\n✓ Downloaded real German Credit dataset (1000 records)")
    logger.info("✓ Prepared data for model training")
    logger.info("✓ Trained XGBoost model on real data")
    logger.info("✓ Model saved and ready to use")
    logger.info("\nNext steps:")
    logger.info("  1. Run dashboard: streamlit run dashboard.py")
    logger.info("  2. Use model in your application")
    logger.info("  3. Download more datasets from Kaggle (see DATA_SOURCES.md)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
