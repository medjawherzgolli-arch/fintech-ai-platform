"""
Quick start script to demonstrate the Fintech AI Platform capabilities.

This script will:
1. Generate synthetic data for all three modules
2. Train models
3. Display performance metrics
4. Save trained models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_generators.credit_risk_generator import CreditRiskDataGenerator
from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator
from src.data_generators.document_processing_generator import DocumentProcessingDataGenerator

from src.credit_risk.model import CreditRiskModel
from src.fraud_detection.model import FraudDetectionModel
from src.document_processing.model import DocumentProcessor

from src.utils.logger import setup_logger

import warnings
warnings.filterwarnings('ignore')


def main():
    """Run the quick start demonstration."""

    # Setup logger
    logger = setup_logger("quickstart", level="INFO")

    logger.info("=" * 70)
    logger.info("FINTECH AI PLATFORM - QUICK START DEMONSTRATION")
    logger.info("=" * 70)

    # ====================================================================
    # 1. CREDIT RISK ASSESSMENT
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("1. CREDIT RISK ASSESSMENT")
    logger.info("=" * 70)

    logger.info("\nGenerating synthetic credit risk data...")
    credit_gen = CreditRiskDataGenerator(random_state=42)
    credit_df = credit_gen.generate(n_samples=10000, default_rate=0.15)
    logger.info(f"Generated {len(credit_df)} loan applications")
    logger.info(f"Default rate: {credit_df['default'].mean():.2%}")

    logger.info("\nTraining credit risk model...")
    credit_model = CreditRiskModel(model_type="xgboost", random_state=42)
    credit_metrics = credit_model.train(credit_df, test_size=0.2, cv_folds=5)

    logger.info("\nCredit Risk Model Performance:")
    logger.info(f"  Accuracy:  {credit_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {credit_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {credit_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {credit_metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {credit_metrics['roc_auc']:.4f}")

    logger.info("\nTop 5 Most Important Features:")
    importance = credit_model.get_feature_importance()
    for idx, row in importance.head(5).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    credit_model.save("models/credit_risk/credit_risk_model.pkl")
    logger.info("\nModel saved to: models/credit_risk/credit_risk_model.pkl")

    # ====================================================================
    # 2. FRAUD DETECTION
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("2. FRAUD DETECTION")
    logger.info("=" * 70)

    logger.info("\nGenerating synthetic transaction data...")
    fraud_gen = FraudDetectionDataGenerator(random_state=42)
    fraud_df = fraud_gen.generate(n_samples=50000, fraud_rate=0.02)
    logger.info(f"Generated {len(fraud_df)} transactions")
    logger.info(f"Fraud rate: {fraud_df['is_fraud'].mean():.2%}")

    logger.info("\nTraining fraud detection model...")
    fraud_model = FraudDetectionModel(model_type="xgboost", random_state=42)
    fraud_metrics = fraud_model.train(fraud_df, test_size=0.2, contamination=0.02)

    logger.info("\nFraud Detection Model Performance:")
    logger.info(f"  Accuracy:  {fraud_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {fraud_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {fraud_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {fraud_metrics['f1_score']:.4f}")
    logger.info(f"  ROC AUC:   {fraud_metrics['roc_auc']:.4f}")

    fraud_model.save("models/fraud_detection/fraud_detection_model.pkl")
    logger.info("\nModel saved to: models/fraud_detection/fraud_detection_model.pkl")

    # ====================================================================
    # 3. DOCUMENT PROCESSING
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("3. DOCUMENT PROCESSING")
    logger.info("=" * 70)

    logger.info("\nGenerating synthetic documents...")
    doc_gen = DocumentProcessingDataGenerator(random_state=42)
    doc_df = doc_gen.generate(n_samples=1000)
    logger.info(f"Generated {len(doc_df)} documents")
    logger.info("\nDocument type distribution:")
    for doc_type, count in doc_df['document_type'].value_counts().items():
        logger.info(f"  {doc_type}: {count}")

    logger.info("\nProcessing documents...")
    processor = DocumentProcessor()
    results = processor.process_batch(doc_df)

    logger.info("\nDocument Processing Results:")
    logger.info(f"  Documents processed: {len(results)}")
    logger.info(f"  Average fields extracted per document: {results['word_count'].mean():.0f} words")

    # Show sample extraction
    sample_doc = doc_df[doc_df['document_type'] == 'invoice'].iloc[0]
    sample_result = processor.process_document(sample_doc['text'], 'invoice')

    logger.info("\nSample Invoice Extraction:")
    logger.info(f"  Invoice Number: {sample_result.get('invoice_number', 'N/A')}")
    logger.info(f"  Total Amount: ${sample_result.get('total_amount', 0):.2f}")
    logger.info(f"  Date Count: {sample_result.get('date_count', 0)}")
    logger.info(f"  Word Count: {sample_result.get('word_count', 0)}")

    processor.save("models/document_processing/document_processor.pkl")
    logger.info("\nProcessor saved to: models/document_processing/document_processor.pkl")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("QUICK START COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("\nNext steps:")
    logger.info("1. Explore the Jupyter notebooks in notebooks/")
    logger.info("2. Check the saved models in models/")
    logger.info("3. Review the comprehensive README.md")
    logger.info("4. Customize config/config.yaml for your use case")
    logger.info("5. Integrate the models into your application")
    logger.info("\nFor more information, see README.md")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
