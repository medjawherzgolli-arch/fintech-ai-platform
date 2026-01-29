# Fintech AI Platform

A comprehensive AI platform for financial services, providing solutions for **credit risk assessment**, **fraud detection**, and **document processing**.

## Features

### 1. Credit Risk Assessment
- Predictive modeling for loan default probability
- Credit scoring and risk classification
- Feature importance analysis
- Support for XGBoost, LightGBM, and CatBoost models

### 2. Fraud Detection
- Real-time transaction monitoring
- Anomaly detection using Isolation Forest and XGBoost
- Pattern recognition for fraudulent activities
- Business metrics for financial impact analysis

### 3. Document Processing
- Text extraction from financial documents
- Automated field extraction (amounts, dates, account numbers)
- Document classification (invoices, statements, tax forms, contracts)
- Named entity recognition for financial entities

## Data Options

### Synthetic Data (Built-in)
The platform includes synthetic data generators for immediate testing and development. No external data needed!

### Real Public Datasets
Want to train on real data? See [DATA_SOURCES.md](DATA_SOURCES.md) for:
- **Credit Risk**: German Credit (UCI), Lending Club, Home Credit (Kaggle)
- **Fraud Detection**: Credit Card Fraud (Kaggle), IEEE-CIS Fraud
- **Documents**: SROIE Receipts, FUNSD Forms, RVL-CDIP

Download real data:
```bash
python scripts/download_real_data.py
```

## Project Structure

```
fintech-ai-platform/
├── src/
│   ├── credit_risk/           # Credit risk modeling
│   │   ├── __init__.py
│   │   └── model.py
│   ├── fraud_detection/       # Fraud detection
│   │   ├── __init__.py
│   │   └── model.py
│   ├── document_processing/   # Document processing
│   │   ├── __init__.py
│   │   └── model.py
│   ├── data_generators/       # Synthetic data generators
│   │   ├── __init__.py
│   │   ├── credit_risk_generator.py
│   │   ├── fraud_detection_generator.py
│   │   └── document_processing_generator.py
│   └── utils/                 # Shared utilities
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── metrics.py
├── notebooks/                 # Jupyter notebooks
│   ├── credit_risk/
│   │   └── 01_credit_risk_analysis.ipynb
│   ├── fraud_detection/
│   └── document_processing/
├── data/                      # Data directory
│   ├── raw/
│   └── processed/
├── models/                    # Saved models
│   ├── credit_risk/
│   ├── fraud_detection/
│   └── document_processing/
├── config/                    # Configuration files
│   └── config.yaml
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd fintech-ai-platform
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package in development mode:
```bash
pip install -e .
```

## Quick Start

### 1. Credit Risk Assessment

```python
from src.data_generators.credit_risk_generator import CreditRiskDataGenerator
from src.credit_risk.model import CreditRiskModel

# Generate synthetic data
generator = CreditRiskDataGenerator(random_state=42)
df = generator.generate(n_samples=10000, default_rate=0.15)

# Train model
model = CreditRiskModel(model_type="xgboost", random_state=42)
metrics = model.train(df, test_size=0.2, cv_folds=5)

# Make predictions
predictions = model.predict(df)
probabilities = model.predict_proba(df)

# Feature importance
importance = model.get_feature_importance()
print(importance)

# Save model
model.save("models/credit_risk/my_model.pkl")
```

### 2. Fraud Detection

```python
from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator
from src.fraud_detection.model import FraudDetectionModel

# Generate synthetic transaction data
generator = FraudDetectionDataGenerator(random_state=42)
df = generator.generate(n_samples=50000, fraud_rate=0.02)

# Train model
model = FraudDetectionModel(model_type="xgboost", random_state=42)
metrics = model.train(df, test_size=0.2, contamination=0.02)

# Detect fraud
fraud_predictions = model.predict(df)
fraud_probabilities = model.predict_proba(df)

# Save model
model.save("models/fraud_detection/my_model.pkl")
```

### 3. Document Processing

```python
from src.data_generators.document_processing_generator import DocumentProcessingDataGenerator
from src.document_processing.model import DocumentProcessor

# Generate synthetic documents
generator = DocumentProcessingDataGenerator(random_state=42)
df = generator.generate(n_samples=1000)

# Initialize processor
processor = DocumentProcessor()

# Process documents
results = processor.process_batch(df, text_column='text')

# Process single document
document_text = df.iloc[0]['text']
extracted_info = processor.process_document(document_text)
print(extracted_info)

# Save processor
processor.save("models/document_processing/processor.pkl")
```

## Using Jupyter Notebooks

Launch Jupyter and explore the interactive notebooks:

```bash
jupyter notebook
```

Navigate to the `notebooks/` directory and open:
- `credit_risk/01_credit_risk_analysis.ipynb` - Credit risk modeling walkthrough
- Additional notebooks for fraud detection and document processing

## Configuration

Edit `config/config.yaml` to customize:
- Data paths
- Model parameters
- Training configurations
- Feature settings

```python
from src.utils.config import get_config

config = get_config()
algorithm = config.get('credit_risk.model.algorithm')
data_path = config.get('data.raw_dir')
```

## Data Generation

All three modules come with synthetic data generators:

### Credit Risk Data
- Borrower demographics (age, income, employment)
- Credit metrics (score, debt-to-income, delinquencies)
- Loan characteristics (amount, term, interest rate)
- Default labels with realistic correlations

### Fraud Detection Data
- Transaction details (amount, time, merchant)
- Customer behavior patterns
- Location and device information
- Fraud labels with realistic patterns

### Document Processing Data
- Invoice documents
- Bank statements
- Tax forms
- Contracts

## Model Performance

### Credit Risk Model
- Accuracy: ~85%
- ROC AUC: ~0.88
- Precision/Recall: Tunable based on business requirements

### Fraud Detection Model
- Detection Rate: ~75-80%
- False Positive Rate: ~1-2%
- ROC AUC: ~0.90

### Document Processor
- Field Extraction Accuracy: ~90%
- Document Classification: ~95%
- Supports multiple document types

## Utilities

### Logging
```python
from src.utils.logger import setup_logger

logger = setup_logger("my_app", level="INFO", log_file="logs/app.log")
logger.info("Processing started")
```

### Metrics
```python
from src.utils.metrics import calculate_classification_metrics

metrics = calculate_classification_metrics(y_true, y_pred, y_pred_proba)
print(f"ROC AUC: {metrics['roc_auc']:.4f}")
```

## Testing

Run tests with pytest:

```bash
pytest tests/
```

## Use Cases

### Credit Risk
- **Loan Approval**: Automate credit decisions based on risk scores
- **Portfolio Management**: Monitor portfolio risk and default probability
- **Pricing**: Risk-based pricing for loans and credit products

### Fraud Detection
- **Transaction Monitoring**: Real-time fraud detection on transactions
- **Account Protection**: Identify compromised accounts
- **Chargeback Prevention**: Reduce fraud-related chargebacks

### Document Processing
- **KYC/AML**: Extract information from identity documents
- **Invoice Processing**: Automate accounts payable/receivable
- **Compliance**: Extract and verify required document fields

## Advanced Features

### Model Comparison
Compare multiple model types:

```python
for model_type in ['xgboost', 'lightgbm', 'catboost']:
    model = CreditRiskModel(model_type=model_type)
    metrics = model.train(df)
    print(f"{model_type} ROC AUC: {metrics['roc_auc']:.4f}")
```

### Custom Features
Add your own features to improve model performance:

```python
# Add custom feature
df['custom_ratio'] = df['loan_amount'] / df['annual_income']

# Train with custom features
model.train(df)
```

### Hyperparameter Tuning
Use scikit-learn's GridSearchCV for optimization:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.3]
}

# Implement tuning for your use case
```

## Performance Optimization

### For Large Datasets
- Use batch processing for predictions
- Enable early stopping in training
- Consider using LightGBM for faster training

### For Production
- Use model serialization for faster loading
- Implement caching for repeated predictions
- Monitor model performance with logging

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation in the `notebooks/` directory

## Roadmap

Future enhancements:
- [ ] Deep learning models for fraud detection
- [ ] Advanced NLP with transformers for document processing
- [ ] REST API for model serving
- [ ] Real-time streaming data support
- [ ] Dashboard for monitoring and visualization
- [ ] Integration with cloud platforms (AWS, Azure, GCP)
- [ ] Explainability features (SHAP, LIME)
- [ ] A/B testing framework
- [ ] Model versioning and experiment tracking

## Acknowledgments

Built with:
- XGBoost, LightGBM, CatBoost for gradient boosting
- Scikit-learn for ML utilities
- Pandas for data manipulation
- PyTorch/TensorFlow for deep learning (future)
- Jupyter for interactive analysis

---

**Note**: This platform generates synthetic data for development and testing. For production use, replace synthetic data with real data and perform thorough validation and compliance checks.
