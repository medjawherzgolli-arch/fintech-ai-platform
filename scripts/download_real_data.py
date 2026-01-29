"""
Download Real Public Datasets for Fintech AI Platform

This script downloads publicly available real datasets for:
1. Credit Risk - Lending Club data, German Credit data
2. Fraud Detection - Credit card fraud datasets
3. Document Processing - Financial document datasets

All datasets are from public, legitimate sources.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import requests
from zipfile import ZipFile
import io

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger

logger = setup_logger("data_download", level="INFO")


def download_file(url: str, output_path: str):
    """Download file from URL."""
    logger.info(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    logger.info(f"Saved to {output_path}")


def download_credit_risk_data():
    """
    Download real credit risk datasets.

    Sources:
    1. German Credit Data (UCI ML Repository)
    2. Lending Club Loan Data (Kaggle/public)
    """
    logger.info("=" * 70)
    logger.info("DOWNLOADING CREDIT RISK DATA")
    logger.info("=" * 70)

    data_dir = Path("data/raw/credit_risk")
    data_dir.mkdir(parents=True, exist_ok=True)

    # German Credit Data from UCI
    logger.info("\n1. German Credit Data (UCI ML Repository)")
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        output_path = data_dir / "german_credit.data"
        download_file(url, output_path)

        # Load and process
        columns = [
            'status_checking', 'duration_months', 'credit_history', 'purpose',
            'credit_amount', 'savings', 'employment_since', 'installment_rate',
            'personal_status', 'other_debtors', 'residence_since', 'property',
            'age', 'other_installments', 'housing', 'existing_credits',
            'job', 'num_dependents', 'telephone', 'foreign_worker', 'credit_risk'
        ]

        df = pd.read_csv(output_path, sep=' ', names=columns, header=None)
        # Convert credit_risk: 1=Good, 2=Bad -> 0=No Default, 1=Default
        df['default'] = (df['credit_risk'] == 2).astype(int)
        df = df.drop('credit_risk', axis=1)

        df.to_csv(data_dir / "german_credit.csv", index=False)
        logger.info(f"Processed {len(df)} records")
        logger.info(f"Default rate: {df['default'].mean():.2%}")

    except Exception as e:
        logger.error(f"Error downloading German Credit Data: {e}")

    # Give Credit Default Data from UCI
    logger.info("\n2. Default of Credit Card Clients (UCI)")
    logger.info("Visit: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients")
    logger.info("Manual download recommended due to license agreement")


def download_fraud_detection_data():
    """
    Download real fraud detection datasets.

    Sources:
    1. Credit Card Fraud Detection (Kaggle - requires API key)
    2. IEEE-CIS Fraud Detection (Kaggle - requires API key)
    """
    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOADING FRAUD DETECTION DATA")
    logger.info("=" * 70)

    data_dir = Path("data/raw/fraud_detection")
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\n1. Credit Card Fraud Detection Dataset")
    logger.info("This dataset requires Kaggle API access")
    logger.info("Steps to download:")
    logger.info("  1. Install kaggle: pip install kaggle")
    logger.info("  2. Setup Kaggle API credentials: https://www.kaggle.com/docs/api")
    logger.info("  3. Run: kaggle datasets download -d mlg-ulb/creditcardfraud")
    logger.info("  4. Extract to: data/raw/fraud_detection/")

    logger.info("\nAlternative: Synthetic Financial Datasets")
    logger.info("You can use the built-in synthetic data generator for testing")


def download_document_data():
    """
    Information about document datasets.

    Document processing datasets are harder to find publicly due to privacy.
    """
    logger.info("\n" + "=" * 70)
    logger.info("DOCUMENT PROCESSING DATA")
    logger.info("=" * 70)

    logger.info("\nPublic financial document datasets are limited due to privacy concerns.")
    logger.info("Options:")
    logger.info("  1. Use synthetic data generator (recommended for testing)")
    logger.info("  2. SROIE Dataset - Receipts/Invoices: https://rrc.cvc.uab.es/?ch=13")
    logger.info("  3. RVL-CDIP Dataset - Document images: https://www.cs.cmu.edu/~aharley/rvl-cdip/")
    logger.info("  4. Create your own dataset with public company annual reports")


def setup_kaggle_instructions():
    """Print instructions for setting up Kaggle."""
    logger.info("\n" + "=" * 70)
    logger.info("KAGGLE SETUP INSTRUCTIONS")
    logger.info("=" * 70)
    logger.info("\nMany quality datasets require Kaggle API access:")
    logger.info("\n1. Install Kaggle CLI:")
    logger.info("   pip install kaggle")
    logger.info("\n2. Get API credentials:")
    logger.info("   - Go to https://www.kaggle.com/settings")
    logger.info("   - Click 'Create New API Token'")
    logger.info("   - Save kaggle.json to ~/.kaggle/ (Linux/Mac) or C:\\Users\\<username>\\.kaggle\\ (Windows)")
    logger.info("\n3. Download datasets:")
    logger.info("   kaggle datasets download -d <dataset-name>")


def main():
    """Main function to download all datasets."""
    logger.info("=" * 70)
    logger.info("FINTECH AI PLATFORM - REAL DATA DOWNLOADER")
    logger.info("=" * 70)
    logger.info("\nThis script will download publicly available real datasets.")
    logger.info("Some datasets may require manual download or API access.")

    # Create data directories
    Path("data/raw/credit_risk").mkdir(parents=True, exist_ok=True)
    Path("data/raw/fraud_detection").mkdir(parents=True, exist_ok=True)
    Path("data/raw/documents").mkdir(parents=True, exist_ok=True)

    # Download datasets
    download_credit_risk_data()
    download_fraud_detection_data()
    download_document_data()
    setup_kaggle_instructions()

    logger.info("\n" + "=" * 70)
    logger.info("DOWNLOAD INFORMATION COMPLETE")
    logger.info("=" * 70)
    logger.info("\nSome datasets require manual download or Kaggle API access.")
    logger.info("See the instructions above for each dataset.")
    logger.info("\nFor immediate testing, use the synthetic data generators:")
    logger.info("  python quickstart.py")


if __name__ == "__main__":
    main()
