# Dashboard Guide

## 🚀 Launch Dashboard

### Option 1: Double-click (Windows)
```
Double-click: launch_dashboard.bat
```

### Option 2: Command Line
```bash
streamlit run dashboard.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

---

## 📊 Dashboard Features

### 1. Overview Page 🏠
- Platform statistics
- Model status
- Quick capabilities summary

### 2. Credit Risk Assessment 💳
**Single Prediction:**
- Adjust borrower details with sliders
- See real-time risk scores
- Visual risk gauge
- Instant approve/reject decisions

**Batch Analysis:**
- Generate 10-1000 applications
- Risk distribution charts
- Approval rate analysis
- Sample data viewer

**Model Insights:**
- Feature importance rankings
- Model performance metrics
- Interactive visualizations

### 3. Fraud Detection 🚨
**Single Transaction:**
- Configure transaction details
- Real-time fraud probability
- Risk factor breakdown
- Block/approve decisions

**Batch Analysis:**
- Analyze up to 5000 transactions
- Fraud vs legitimate comparisons
- Time-based fraud patterns
- Interactive charts

### 4. Document Processing 📄
**Single Document:**
- Paste invoice/statement text
- Extract structured data
- View all extracted fields
- JSON output

**Batch Processing:**
- Process up to 500 documents
- Document type distribution
- Amount analysis
- Data export

### 5. Model Performance 📊
- Complete metrics for all models
- Accuracy, precision, recall, ROC AUC
- Comparison views
- Model information

---

## 🎮 How to Use

### Test Credit Risk Assessment
1. Go to "Credit Risk Assessment" page
2. Click "Single Prediction" tab
3. Adjust sliders for a borrower profile
4. Click "Assess Risk"
5. See instant risk score and decision!

### Test Fraud Detection
1. Go to "Fraud Detection" page
2. Enter transaction details
3. Click "Check for Fraud"
4. See fraud probability and risk factors!

### Process Documents
1. Go to "Document Processing" page
2. Paste an invoice or statement
3. Click "Process Document"
4. See extracted amounts, dates, emails!

---

## 💡 Tips

### Visualize Everything
- Use batch analysis to see patterns
- Charts update in real-time
- Export data for further analysis

### Compare Models
- Test same profile on synthetic vs real data model
- See how model performance differs
- Understand real-world vs perfect scenarios

### Interactive Testing
- All sliders update live
- No need to reload
- Instant feedback

---

## 🔧 Troubleshooting

### Dashboard won't start?
```bash
pip install streamlit plotly
streamlit run dashboard.py
```

### Port already in use?
```bash
streamlit run dashboard.py --server.port 8502
```

### Can't see charts?
```bash
pip install plotly --upgrade
```

---

## 📈 What You're Seeing

### Real-Time AI Predictions
- Every slider change triggers model
- Live probability calculations
- Instant risk assessments

### Interactive Visualizations
- Risk gauges update live
- Charts respond to your inputs
- Explore data visually

### Business Insights
- Not just ML metrics
- Approval rates, amounts, patterns
- Real business decisions

---

## 🎯 Next Steps

### After Dashboard Exploration:
1. **Download more real data** - See DATA_SOURCES.md
2. **Train on your data** - Modify get_real_data.py
3. **Integrate into app** - Use models via API
4. **Deploy to production** - See deployment guides

### Want More Features?
- Add API endpoints for models
- Create mobile app interface
- Build automated reporting
- Add model retraining pipelines

---

## 📞 Need Help?

- Check README.md for platform overview
- See DATA_SOURCES.md for dataset options
- Review notebooks/ for detailed examples
- Run simple_test.py for quick tests

---

**🚀 Enjoy your AI-powered financial analysis dashboard!**
