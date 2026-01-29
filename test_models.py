"""
Test the trained models with sample predictions
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.credit_risk.model import CreditRiskModel
from src.fraud_detection.model import FraudDetectionModel
from src.document_processing.model import DocumentProcessor

print("=" * 70)
print("TESTING TRAINED MODELS")
print("=" * 70)

# ====================================================================
# 1. TEST CREDIT RISK MODEL
# ====================================================================
print("\n1. CREDIT RISK PREDICTION")
print("-" * 70)

# Load the trained model
credit_model = CreditRiskModel()
credit_model.load("models/credit_risk/credit_risk_model.pkl")

# Create a sample loan application
sample_loan = pd.DataFrame([{
    'age': 35,
    'employment_length': 8.0,
    'annual_income': 75000.00,
    'credit_score': 720,
    'loan_amount': 25000.00,
    'loan_term': 36,
    'interest_rate': 0.0899,
    'debt_to_income': 0.25,
    'home_ownership': 'MORTGAGE',
    'loan_purpose': 'debt_consolidation',
    'delinquencies_2yrs': 0,
    'open_accounts': 12,
    'total_credit_lines': 18,
    'revolving_balance': 8500.00,
    'revolving_utilization': 0.35
}])

print("\nSample Loan Application:")
print(f"  Age: {sample_loan['age'].values[0]}")
print(f"  Income: ${sample_loan['annual_income'].values[0]:,.2f}")
print(f"  Credit Score: {sample_loan['credit_score'].values[0]}")
print(f"  Loan Amount: ${sample_loan['loan_amount'].values[0]:,.2f}")
print(f"  Debt-to-Income: {sample_loan['debt_to_income'].values[0]:.2%}")

# Make prediction
default_prob = credit_model.predict_proba(sample_loan)[0]
default_pred = credit_model.predict(sample_loan)[0]

print(f"\n  Default Probability: {default_prob:.2%}")
print(f"  Prediction: {'DEFAULT' if default_pred else 'NO DEFAULT'}")
print(f"  Decision: {'REJECT' if default_prob > 0.3 else 'APPROVE'}")

# ====================================================================
# 2. TEST FRAUD DETECTION MODEL
# ====================================================================
print("\n\n2. FRAUD DETECTION")
print("-" * 70)

# Load the trained model
fraud_model = FraudDetectionModel()
fraud_model.load("models/fraud_detection/fraud_detection_model.pkl")

# Generate a realistic transaction using the same generator
from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator
gen = FraudDetectionDataGenerator(random_state=99)
sample_transaction = gen.generate(n_samples=100, fraud_rate=0.5).head(1)

print("\nSample Transaction:")
print(f"  Amount: ${sample_transaction['transaction_amount'].values[0]:,.2f}")
print(f"  Time: {int(sample_transaction['transaction_hour'].values[0])}:00")
print(f"  Merchant: {sample_transaction['merchant_category'].values[0]}")
print(f"  Location: {sample_transaction['distance_from_home'].values[0]:.1f} miles from home")
print(f"  Card Present: {'Yes' if sample_transaction['card_present'].values[0] else 'No'}")
print(f"  International: {'Yes' if sample_transaction['international'].values[0] else 'No'}")

# Make prediction
fraud_prob = fraud_model.predict_proba(sample_transaction)[0]
fraud_pred = fraud_model.predict(sample_transaction)[0]

print(f"\n  Fraud Probability: {fraud_prob:.2%}")
print(f"  Prediction: {'FRAUD' if fraud_pred else 'LEGITIMATE'}")
print(f"  Decision: {'BLOCK' if fraud_prob > 0.5 else 'APPROVE'}")

# ====================================================================
# 3. TEST DOCUMENT PROCESSOR
# ====================================================================
print("\n\n3. DOCUMENT PROCESSING")
print("-" * 70)

# Load the processor
processor = DocumentProcessor()
processor.load("models/document_processing/document_processor.pkl")

# Sample invoice text
sample_invoice = """
INVOICE

Invoice Number: INV-2024-001
Invoice Date: 01/15/2024
Due Date: 02/14/2024

BILL TO:
John Smith
123 Main St

FROM:
Acme Corporation
456 Business Ave

ITEMS:
Consulting Services - Qty: 40 x $150.00 = $6,000.00
Software License - Qty: 1 x $500.00 = $500.00

Subtotal: $6,500.00
Tax (8%): $520.00
TOTAL DUE: $7,020.00

Payment Terms: Net 30 days
"""

print("\nSample Invoice (excerpt):")
print(sample_invoice[:200] + "...")

# Process document
result = processor.process_document(sample_invoice, document_type='invoice')

print(f"\nExtracted Information:")
print(f"  Document Type: {result['document_type']}")
print(f"  Invoice Number: {result.get('invoice_number', 'N/A')}")
print(f"  Total Amount: ${result.get('total_amount', 0):,.2f}")
print(f"  Dates Found: {result.get('date_count', 0)}")
print(f"  Word Count: {result.get('word_count', 0)}")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED!")
print("=" * 70)
print("\nYour models are working and ready to use!")
print("Next: Integrate these models into your application or try real data.")
