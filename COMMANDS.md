# 🚀 Platform Commands Reference

Quick reference for all commands to use your Fintech AI Platform.

---

## 📊 Dashboard (Main Interface)

### Launch Interactive Dashboard:
```bash
cd fintech-ai-platform
streamlit run dashboard.py
```
**Opens at:** http://localhost:8501

This is your main interface with:
- Real-time predictions
- Interactive visualizations
- Model switching (synthetic vs real)
- Batch analysis

---

## 🧪 Testing & Demos

### Quick Test (All Models):
```bash
cd fintech-ai-platform
python quickstart.py
```
Runs complete demo of all 3 models.

### Simple Test:
```bash
cd fintech-ai-platform
python simple_test.py
```
Quick credit risk + document processing test.

### Test with Real Data:
```bash
cd fintech-ai-platform
python test_models.py
```
Comprehensive model testing.

---

## 📥 Data Management

### Download ALL Real Data:
```bash
cd fintech-ai-platform
python download_all_real_data.py
```
Downloads and trains on:
- German Credit (1K loans)
- Kaggle Credit Card Fraud (284K transactions)

### Download Specific Dataset:
```bash
cd fintech-ai-platform
python get_real_data.py
```
Downloads German Credit only.

### Check What Data You Have:
```bash
cd fintech-ai-platform
dir data\raw /s
```
or
```bash
ls -R data/raw/
```

---

## 🤖 Model Operations

### Train Credit Risk Model:
```python
cd fintech-ai-platform
python -c "from src.credit_risk.model import CreditRiskModel; model = CreditRiskModel(); model.train(your_data)"
```

### Load and Use Model:
```python
python
>>> from src.credit_risk.model import CreditRiskModel
>>> model = CreditRiskModel()
>>> model.load("models/credit_risk/credit_risk_model_real_data.pkl")
>>> # Now use model.predict(data)
```

### Check Model Files:
```bash
cd fintech-ai-platform
dir models /s /b
```
or
```bash
ls -lh models/*/*.pkl
```

---

## 📓 Jupyter Notebooks

### Launch Jupyter:
```bash
cd fintech-ai-platform
jupyter notebook
```

Opens notebook interface. Navigate to `notebooks/` folder.

### Run Specific Notebook:
```bash
cd fintech-ai-platform
jupyter notebook notebooks/credit_risk/01_credit_risk_analysis.ipynb
```

---

## 🔧 Installation & Setup

### Install All Dependencies:
```bash
cd fintech-ai-platform
pip install -r requirements.txt
```

### Install Specific Package:
```bash
pip install streamlit
pip install kaggle
pip install xgboost
```

### Check Python Version:
```bash
python --version
```

### Check Installed Packages:
```bash
pip list
```

---

## 📊 Kaggle (For More Data)

### Setup Kaggle:
```bash
pip install kaggle
# Then add API key to: C:\Users\<username>\.kaggle\kaggle.json
```

### Download Dataset:
```bash
kaggle datasets download -d mlg-ulb/creditcardfraud
```

### List Available Datasets:
```bash
kaggle datasets list -s "credit risk"
```

---

## 📁 File Management

### Navigate to Project:
```bash
cd c:\Users\medja\OneDrive\Documents\Programs\fintech-ai-platform
```

### List Files:
```bash
dir          # Windows
ls           # Linux/Mac
```

### View File Contents:
```bash
type README.md          # Windows
cat README.md           # Linux/Mac
```

### Open in VS Code:
```bash
code .
```

---

## 🚀 Running in Background

### Run Dashboard in Background:
```bash
start streamlit run dashboard.py
```

### Stop Background Process:
- Press Ctrl+C in terminal
- Or close the terminal window

---

## 🐛 Troubleshooting

### Dashboard Won't Start?
```bash
pip install streamlit plotly --upgrade
streamlit run dashboard.py --server.port 8502
```

### Module Not Found Error?
```bash
cd fintech-ai-platform
pip install -r requirements.txt
```

### Permission Denied?
```bash
# Run terminal as Administrator (Windows)
# Or use sudo (Linux/Mac)
```

### Check if Port is in Use:
```bash
netstat -ano | findstr :8501     # Windows
lsof -i :8501                    # Linux/Mac
```

---

## 📖 Documentation Commands

### View README:
```bash
cd fintech-ai-platform
type README.md          # Windows
cat README.md           # Linux/Mac
```

### View Available Data:
```bash
type DATA_SOURCES.md
```

### View Model Status:
```bash
type MODEL_STATUS.md
```

### View Real Data Info:
```bash
type REAL_DATA_COMPLETE.md
```

---

## 💡 Quick Start Workflow

**First Time Setup:**
```bash
cd fintech-ai-platform
pip install -r requirements.txt
python download_all_real_data.py
```

**Daily Use:**
```bash
cd fintech-ai-platform
streamlit run dashboard.py
# Opens at http://localhost:8501
```

**Testing:**
```bash
python quickstart.py
# or
python simple_test.py
```

---

## 🎯 Most Used Commands

These are the commands you'll use 90% of the time:

```bash
# 1. Start Dashboard (Most Important!)
cd fintech-ai-platform
streamlit run dashboard.py

# 2. Quick Test
python simple_test.py

# 3. Download More Data
python download_all_real_data.py

# 4. Check What You Have
dir models /s /b
dir data\raw /s
```

---

## ❌ Commands That DON'T Exist

**These will NOT work:**
```bash
claude              # No such command
fintech            # No such command
ai-platform        # No such command
train-model        # No such command
```

**Instead use:**
```bash
streamlit run dashboard.py    # For dashboard
python quickstart.py          # For demos
python simple_test.py         # For tests
```

---

## 🆘 Getting Help

### Platform Help:
```bash
type README.md
type COMMANDS.md         # This file!
```

### Python Help:
```bash
python -c "from src.credit_risk.model import CreditRiskModel; help(CreditRiskModel)"
```

### Streamlit Help:
```bash
streamlit --help
```

---

## 📱 VS Code Integration

You're using **Claude in VS Code** - that's me! I'm your AI assistant helping you build this.

**You don't need to run "claude" as a command** - I'm already helping you right here in the VS Code extension.

**Your platform commands:**
- `streamlit run dashboard.py` - Your dashboard
- `python quickstart.py` - Your demos
- `python simple_test.py` - Your tests

---

## ✅ Summary

**What You Built:**
- Fintech AI Platform (your project)
- 3 AI models (credit, fraud, documents)
- Interactive dashboard (Streamlit)

**How to Use It:**
```bash
cd fintech-ai-platform
streamlit run dashboard.py
# Opens at http://localhost:8501
```

**Who is Claude:**
- I'm your AI assistant in VS Code
- I helped you build this
- No separate command needed!

---

**🚀 Start your dashboard now:**
```bash
cd fintech-ai-platform
streamlit run dashboard.py
```

**Then open:** http://localhost:8501
