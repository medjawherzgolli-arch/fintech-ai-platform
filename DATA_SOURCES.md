# Real Data Sources for Fintech AI Platform

This document lists publicly available real datasets you can use with this platform.

## ⚠️ Important Notes

- **Privacy**: Only use datasets you have permission to use
- **Compliance**: Ensure compliance with GDPR, CCPA, and other regulations
- **Licensing**: Check dataset licenses before commercial use
- **Anonymization**: Most public datasets are already anonymized

---

## 🏦 Credit Risk Datasets

### 1. German Credit Data ✅ FREE
- **Source**: UCI Machine Learning Repository
- **Records**: 1,000 loan applications
- **Features**: 20+ features including credit history, purpose, age, employment
- **Target**: Credit risk (good/bad)
- **Download**: Automatic via script
- **URL**: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)

```bash
python scripts/download_real_data.py
```

### 2. Default of Credit Card Clients ✅ FREE
- **Source**: UCI Machine Learning Repository
- **Records**: 30,000 credit card clients
- **Features**: Payment history, bill amounts, age, education
- **Target**: Default next month
- **URL**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients

### 3. Lending Club Loan Data 🔑 KAGGLE
- **Source**: Kaggle
- **Records**: 2+ million loans (2007-2018)
- **Features**: Comprehensive loan and borrower data
- **Target**: Loan status (charged off, fully paid, etc.)
- **URL**: https://www.kaggle.com/datasets/wordsforthewise/lending-club

```bash
kaggle datasets download -d wordsforthewise/lending-club
```

### 4. Give Me Some Credit 🔑 KAGGLE
- **Source**: Kaggle Competition
- **Records**: 150,000 borrowers
- **Features**: Credit utilization, delinquencies, income
- **Target**: Financial distress
- **URL**: https://www.kaggle.com/c/GiveMeSomeCredit

### 5. Home Credit Default Risk 🔑 KAGGLE
- **Source**: Kaggle Competition (2018)
- **Records**: 307,511 applications
- **Features**: Multiple data tables (application, bureau, previous applications)
- **Target**: Repayment difficulties
- **URL**: https://www.kaggle.com/c/home-credit-default-risk

---

## 💳 Fraud Detection Datasets

### 1. Credit Card Fraud Detection 🔑 KAGGLE ⭐ RECOMMENDED
- **Source**: Kaggle (from ULB)
- **Records**: 284,807 transactions
- **Features**: 30 features (PCA transformed for privacy)
- **Fraud Rate**: 0.172%
- **URL**: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

### 2. IEEE-CIS Fraud Detection 🔑 KAGGLE
- **Source**: Kaggle Competition (Vesta Corporation)
- **Records**: 590,540 transactions
- **Features**: Transaction data, identity information
- **Fraud Rate**: ~3.5%
- **URL**: https://www.kaggle.com/c/ieee-fraud-detection

### 3. Synthetic Financial Fraud 🔑 KAGGLE
- **Source**: Kaggle (PaySim simulator)
- **Records**: 6+ million transactions
- **Features**: Transaction type, amount, balances
- **URL**: https://www.kaggle.com/datasets/ealaxi/paysim1

### 4. Bank Account Fraud 🔑 KAGGLE
- **Source**: Kaggle
- **Records**: 1 million accounts
- **Features**: Account characteristics, behavior patterns
- **URL**: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

---

## 📄 Document Processing Datasets

### 1. SROIE Dataset ✅ FREE
- **Source**: ICDAR 2019 Competition
- **Type**: Receipt/Invoice OCR and Information Extraction
- **Records**: 1,000 scanned receipts
- **Tasks**: Text detection, recognition, key information extraction
- **URL**: https://rrc.cvc.uab.es/?ch=13

### 2. FUNSD Dataset ✅ FREE
- **Source**: Form Understanding in Noisy Scanned Documents
- **Records**: 199 real forms
- **Features**: Annotated text, labels, relationships
- **URL**: https://guillaumejaume.github.io/FUNSD/

### 3. RVL-CDIP Dataset ✅ FREE
- **Source**: Ryerson Vision Lab
- **Records**: 400,000 document images
- **Classes**: 16 document types (letter, form, invoice, etc.)
- **Size**: ~40 GB
- **URL**: https://www.cs.cmu.edu/~aharley/rvl-cdip/

### 4. DocBank ✅ FREE
- **Source**: Microsoft Research
- **Records**: 500K+ document pages
- **Type**: Document layout analysis
- **URL**: https://doc-analysis.github.io/docbank-page/

### 5. Invoice Dataset 🔑 KAGGLE
- **Source**: Various Kaggle datasets
- **Search**: "invoice dataset" on Kaggle
- **Note**: Multiple smaller datasets available

---

## 🔧 How to Use Real Data

### Step 1: Download Data

```bash
# Create scripts directory if it doesn't exist
mkdir -p scripts

# Run the download script
python scripts/download_real_data.py
```

### Step 2: Setup Kaggle (for 🔑 datasets)

1. Install Kaggle CLI:
```bash
pip install kaggle
```

2. Get API credentials:
   - Go to https://www.kaggle.com/settings
   - Click "Create New API Token"
   - Save `kaggle.json` to:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<username>\.kaggle\kaggle.json`

3. Download datasets:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/fraud_detection/
```

### Step 3: Use with Platform

```python
import pandas as pd
from src.credit_risk.model import CreditRiskModel

# Load real data
df = pd.read_csv("data/raw/credit_risk/german_credit.csv")

# Train model on real data
model = CreditRiskModel(model_type="xgboost")
metrics = model.train(df, test_size=0.2)

print(f"ROC AUC: {metrics['roc_auc']:.4f}")
```

---

## 📊 Recommended Datasets by Use Case

### For Learning & Testing
1. **German Credit Data** - Small, easy to start
2. **Credit Card Fraud** - Classic fraud detection dataset
3. **SROIE Receipts** - Document extraction basics

### For Realistic Development
1. **Lending Club** - Large-scale credit risk
2. **IEEE-CIS Fraud** - Real-world fraud patterns
3. **FUNSD** - Complex document forms

### For Production Simulation
1. **Home Credit Default** - Multiple data tables, realistic complexity
2. **Bank Account Fraud** - Large scale, modern features
3. **RVL-CDIP** - Large-scale document classification

---

## 🌐 Additional Data Sources

### Financial Data Aggregators
- **Kaggle**: https://www.kaggle.com/datasets (1000+ finance datasets)
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/index.php
- **Google Dataset Search**: https://datasetsearch.research.google.com/
- **Data.gov**: https://data.gov/ (government data)
- **World Bank Open Data**: https://data.worldbank.org/

### Academic Sources
- **ICDAR Competitions**: Document analysis datasets
- **NeurIPS Datasets**: ML competition datasets
- **Papers with Code**: Datasets from research papers

### Commercial (Paid)
- **Refinitiv**: Financial market data
- **Bloomberg**: Financial data APIs
- **Experian**: Credit data (B2B)
- **Equifax**: Credit bureau data (B2B)

---

## ⚖️ Legal & Ethical Considerations

### Before Using Any Dataset:

1. **Read the License**: Ensure you can use it for your purpose
2. **Check Terms of Use**: Some datasets prohibit commercial use
3. **Verify Privacy Compliance**: Ensure GDPR/CCPA compliance
4. **Cite Properly**: Give credit to dataset creators
5. **Respect Restrictions**: Don't redistribute if prohibited

### Data Privacy Best Practices:

- Never use real customer data without consent
- Anonymize any sensitive information
- Implement data access controls
- Document data lineage
- Follow industry regulations (PCI-DSS, SOC2, etc.)

---

## 🚀 Quick Start with Real Data

```bash
# 1. Download German Credit Data (automatic, no login needed)
python scripts/download_real_data.py

# 2. Train model on real data
python examples/train_on_real_data.py

# 3. Compare with synthetic data
python examples/compare_synthetic_vs_real.py
```

---

## 📧 Need More Data?

If you need specific datasets:
1. Check Kaggle competitions in finance
2. Look for research papers and their associated datasets
3. Consider partnering with financial institutions
4. Use data augmentation on existing datasets
5. Generate high-quality synthetic data

For production use, you'll eventually need your own proprietary data for the best results.
