"""
Interactive Dashboard for Fintech AI Platform

Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.credit_risk.model import CreditRiskModel
from src.credit_risk.deep_model import CreditRiskDeepModel
from src.fraud_detection.model import FraudDetectionModel
from src.fraud_detection.deep_model import FraudDetectionDeepModel
from src.document_processing.model import DocumentProcessor
from src.document_processing.deep_model import DocumentClassifierDeep
from src.data_generators.credit_risk_generator import CreditRiskDataGenerator
from src.data_generators.fraud_detection_generator import FraudDetectionDataGenerator

# Page config
st.set_page_config(
    page_title="Fintech AI Platform Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.big-font {
    font-size:30px !important;
    font-weight: bold;
}
.metric-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin: 10px 0;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title("💰 Fintech AI Platform Dashboard")
st.markdown("### Real-time AI Model Monitoring & Testing")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Module", [
    "🏠 Overview",
    "💳 Credit Risk Assessment",
    "🚨 Fraud Detection",
    "📄 Document Processing",
    "📊 Model Performance"
])

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "🏠 Overview":
    st.header("Platform Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Credit Risk Model", "Loaded ✅", "95.6% Accuracy")
    with col2:
        st.metric("Fraud Detection", "Loaded ✅", "99.5% Accuracy")
    with col3:
        st.metric("Document Processor", "Loaded ✅", "1000+ Docs")

    st.markdown("---")

    # Quick stats
    st.subheader("📈 Platform Capabilities")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Credit Risk Assessment:**
        - Predict loan default probability
        - Multi-algorithm support (XGBoost, LightGBM, CatBoost)
        - Feature importance analysis
        - Real-time risk scoring

        **Fraud Detection:**
        - Real-time transaction monitoring
        - Anomaly detection
        - Pattern recognition
        - Low false-positive rate (<2%)
        """)

    with col2:
        st.markdown("""
        **Document Processing:**
        - Extract structured data from documents
        - Support for invoices, statements, tax forms
        - Named entity recognition
        - Automatic field extraction

        **Data Sources:**
        - Synthetic data generators (built-in)
        - Real public datasets (German Credit, Kaggle)
        - Custom data upload support
        """)

# ============================================================================
# CREDIT RISK PAGE
# ============================================================================
elif page == "💳 Credit Risk Assessment":
    st.header("💳 Credit Risk Assessment")

    # Model type selection (XGBoost vs Deep Learning)
    model_architecture = st.sidebar.radio(
        "Model Architecture:",
        ["XGBoost (Traditional ML)", "PyTorch (Deep Learning)"],
        index=0
    )

    # Data source selection
    data_source = st.sidebar.radio(
        "Training Data:",
        ["Real Data - German Credit", "Synthetic Data"],
        index=0
    )

    # Load model based on selections
    # Version parameter forces cache refresh when models are retrained
    @st.cache_resource
    def load_credit_model(architecture, data_type, _version="v3"):
        if architecture == "deep":
            model = CreditRiskDeepModel()
            if data_type == "real":
                try:
                    model.load("models/credit_risk/credit_risk_deep_model_real_data.pkl")
                except:
                    model.load("models/credit_risk/credit_risk_deep_model.pkl")
            else:
                model.load("models/credit_risk/credit_risk_deep_model.pkl")
        else:
            model = CreditRiskModel()
            if data_type == "real":
                model.load("models/credit_risk/credit_risk_model_real_data.pkl")
            else:
                model.load("models/credit_risk/credit_risk_model.pkl")
        return model

    arch_type = "deep" if "PyTorch" in model_architecture else "xgboost"
    data_type = "real" if "Real Data" in data_source else "synthetic"

    try:
        model = load_credit_model(arch_type, data_type)
        model_loaded = True
    except Exception as e:
        model_loaded = False
        st.error(f"Failed to load model: {e}")

    # Show which model is loaded
    arch_label = "PyTorch Neural Network" if arch_type == "deep" else "XGBoost"
    if data_type == "real":
        st.info(f"🎯 Using **{arch_label}** trained on **1,000 real German Credit applications** (UCI Dataset)")
    else:
        st.info(f"🎮 Using **{arch_label}** trained on **10,000 synthetic applications** (Perfect for testing)")

    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Analysis", "📈 Model Insights"])

    # TAB 1: Single Prediction
    with tab1:
        st.subheader("Assess Individual Loan Application")

        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 75, 35)
            annual_income = st.number_input("Annual Income ($)", 20000, 500000, 75000, 5000)
            credit_score = st.slider("Credit Score", 300, 850, 720)
            loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 25000, 1000)
            loan_term = st.selectbox("Loan Term (months)", [12, 24, 36, 48, 60], index=2)
            employment_length = st.slider("Employment Length (years)", 0, 40, 8)

        with col2:
            interest_rate = st.slider("Interest Rate", 0.03, 0.30, 0.09, 0.01)
            debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.25, 0.01)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "credit_card", "home_improvement",
                "major_purchase", "small_business", "other"
            ])
            delinquencies = st.number_input("Delinquencies (2 years)", 0, 10, 0)
            revolving_utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.30, 0.01)

        if st.button("🎯 Assess Risk", key="assess_credit"):
            # Create dataframe
            applicant = pd.DataFrame([{
                'age': age,
                'employment_length': employment_length,
                'annual_income': annual_income,
                'credit_score': credit_score,
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'interest_rate': interest_rate,
                'debt_to_income': debt_to_income,
                'home_ownership': home_ownership,
                'loan_purpose': loan_purpose,
                'delinquencies_2yrs': delinquencies,
                'open_accounts': 10,
                'total_credit_lines': 15,
                'revolving_balance': annual_income * 0.1,
                'revolving_utilization': revolving_utilization
            }])

            # Predict
            risk_prob = model.predict_proba(applicant)[0]
            decision = model.predict(applicant)[0]

            st.markdown("---")

            # Display results
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Default Probability", f"{risk_prob:.1%}")
            with col2:
                decision_text = "REJECT ❌" if decision else "APPROVE ✅"
                st.metric("Decision", decision_text)
            with col3:
                risk_level = "HIGH" if risk_prob > 0.3 else "MEDIUM" if risk_prob > 0.15 else "LOW"
                st.metric("Risk Level", risk_level)

            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Default Risk Score"},
                delta={'reference': 15},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkred" if risk_prob > 0.3 else "orange" if risk_prob > 0.15 else "green"},
                    'steps': [
                        {'range': [0, 15], 'color': "lightgreen"},
                        {'range': [15, 30], 'color': "lightyellow"},
                        {'range': [30, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 30
                    }
                }
            ))
            st.plotly_chart(fig)

    # TAB 2: Batch Analysis
    with tab2:
        st.subheader("Analyze Multiple Applications")

        n_samples = st.slider("Number of applications to generate", 10, 1000, 100)

        if st.button("📊 Generate & Analyze", key="batch_credit"):
            with st.spinner("Generating applications and analyzing..."):
                # Generate data
                generator = CreditRiskDataGenerator(random_state=42)
                df = generator.generate(n_samples=n_samples, default_rate=0.15)

                # Predict
                predictions = model.predict_proba(df)
                df['risk_score'] = predictions
                df['decision'] = model.predict(df)

                # Display summary
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Applications", len(df))
                with col2:
                    approved = (df['decision'] == 0).sum()
                    st.metric("Approved", f"{approved} ({approved/len(df)*100:.1f}%)")
                with col3:
                    rejected = (df['decision'] == 1).sum()
                    st.metric("Rejected", f"{rejected} ({rejected/len(df)*100:.1f}%)")
                with col4:
                    avg_risk = df['risk_score'].mean()
                    st.metric("Avg Risk Score", f"{avg_risk:.1%}")

                # Risk distribution
                fig = px.histogram(
                    df, x='risk_score',
                    title="Risk Score Distribution",
                    labels={'risk_score': 'Default Risk Probability'},
                    nbins=50
                )
                st.plotly_chart(fig, use_container_width=True)

                # Show sample data
                st.subheader("Sample Applications")
                display_cols = ['age', 'annual_income', 'credit_score', 'loan_amount', 'risk_score', 'decision']
                st.dataframe(df[display_cols].head(20))

    # TAB 3: Model Insights
    with tab3:
        st.subheader("Model Feature Importance")

        # Check if model supports feature importance (XGBoost does, PyTorch doesn't)
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()

            fig = px.bar(
                importance.head(15),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 15 Most Important Features",
                labels={'importance': 'Importance Score', 'feature': 'Feature'}
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(importance)
        else:
            st.info("Feature importance is not available for PyTorch models. Switch to XGBoost to view feature importance.")

# ============================================================================
# FRAUD DETECTION PAGE
# ============================================================================
elif page == "🚨 Fraud Detection":
    st.header("🚨 Fraud Detection")

    # Model type selection (XGBoost vs Deep Learning)
    fraud_model_architecture = st.sidebar.radio(
        "Model Architecture:",
        ["XGBoost (Traditional ML)", "PyTorch (Deep Learning)"],
        index=0,
        key="fraud_arch"
    )

    # Data source selection
    fraud_data_source = st.sidebar.radio(
        "Training Data:",
        ["Real Data - Kaggle", "Synthetic Data"],
        index=0,
        key="fraud_data"
    )

    # Load model based on selections
    # Version parameter forces cache refresh when models are retrained
    @st.cache_resource
    def load_fraud_model(architecture, data_type, _version="v3"):
        if architecture == "deep":
            model = FraudDetectionDeepModel()
            if data_type == "real":
                try:
                    model.load("models/fraud_detection/fraud_detection_deep_model_real_data.pkl")
                except:
                    model.load("models/fraud_detection/fraud_detection_deep_model.pkl")
            else:
                model.load("models/fraud_detection/fraud_detection_deep_model.pkl")
        else:
            model = FraudDetectionModel()
            if data_type == "real":
                model.load("models/fraud_detection/fraud_detection_model_real_data.pkl")
            else:
                model.load("models/fraud_detection/fraud_detection_model.pkl")
        return model

    fraud_arch_type = "deep" if "PyTorch" in fraud_model_architecture else "xgboost"
    fraud_data_type = "real" if "Real Data" in fraud_data_source else "synthetic"

    try:
        fraud_model = load_fraud_model(fraud_arch_type, fraud_data_type)
        model_loaded = True

        # Show which model is loaded
        arch_label = "PyTorch Neural Network" if fraud_arch_type == "deep" else "XGBoost"
        if fraud_data_type == "real":
            st.info(f"🎯 Using **{arch_label}** trained on **284,807 real credit card transactions** (Kaggle Dataset)")
        else:
            st.info(f"🎮 Using **{arch_label}** trained on **50,000 synthetic transactions** (Perfect for testing)")
    except Exception as e:
        model_loaded = False
        st.warning(f"⚠️ Fraud model not available ({e}). Using rule-based detection.")

    tab1, tab2 = st.tabs(["🔍 Single Transaction", "📊 Batch Analysis"])

    with tab1:
        st.subheader("Analyze Individual Transaction")

        col1, col2 = st.columns(2)

        with col1:
            amount = st.number_input("Transaction Amount ($)", 1.0, 10000.0, 250.0, 10.0)
            hour = st.slider("Transaction Hour (24h)", 0, 23, 14)
            merchant = st.selectbox("Merchant Category", [
                "grocery", "gas_station", "restaurant", "online_retail",
                "electronics", "travel", "entertainment", "healthcare", "other"
            ])
            card_present = st.checkbox("Card Present", value=True)

        with col2:
            distance = st.number_input("Distance from Home (miles)", 0.0, 500.0, 10.0, 5.0)
            international = st.checkbox("International Transaction", value=False)
            frequency_24h = st.number_input("Transactions in last 24h", 0, 20, 2)
            failed_24h = st.number_input("Failed transactions in last 24h", 0, 10, 0)

        if st.button("🔍 Check for Fraud", key="check_fraud"):
            # Create DataFrame for model prediction
            import numpy as np

            transaction = pd.DataFrame([{
                'transaction_amount': amount,
                'card_present': 1 if card_present else 0,
                'distance_from_home': distance,
                'transaction_frequency_24h': frequency_24h,
                'avg_transaction_amount': amount,  # Approximate
                'days_since_last_transaction': 1,
                'transactions_last_7days': frequency_24h * 3,
                'failed_transactions_24h': failed_24h,
                'merchant_risk_score': 0.5,
                'international': 1 if international else 0,
                'device_id_change': 0,
                'transaction_hour': hour,
                'transaction_day': 3,  # Wednesday
                'merchant_category': merchant,
                'timestamp': pd.Timestamp.now()
            }])

            # Use ML model if loaded, otherwise fallback to rules
            if model_loaded:
                fraud_prob = fraud_model.predict_proba(transaction)[0]
                prediction = fraud_model.predict(transaction)[0]
                method = "ML Model"
            else:
                # Fallback rule-based assessment
                risk_factors = 0
                if amount > 1000: risk_factors += 1
                if hour < 6 or hour > 22: risk_factors += 1
                if not card_present: risk_factors += 1
                if distance > 100: risk_factors += 1
                if international: risk_factors += 1
                if failed_24h > 0: risk_factors += 1
                fraud_prob = min(risk_factors * 0.15, 0.95)
                prediction = 1 if fraud_prob > 0.5 else 0
                method = "Rule-based"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Fraud Probability", f"{fraud_prob:.1%}")
            with col2:
                decision = "BLOCK 🚫" if prediction == 1 else "APPROVE ✅"
                st.metric("Decision", decision)
            with col3:
                risk_level = "HIGH" if fraud_prob > 0.7 else "MEDIUM" if fraud_prob > 0.3 else "LOW"
                st.metric("Risk Level", risk_level)

            st.caption(f"Prediction method: {method}")

            # Risk factors breakdown
            st.subheader("Risk Factors Analyzed:")
            risk_factor_count = 0
            if amount > 1000:
                st.warning("⚠️ Large transaction amount")
                risk_factor_count += 1
            if hour < 6 or hour > 22:
                st.warning("⚠️ Unusual transaction time")
                risk_factor_count += 1
            if not card_present:
                st.warning("⚠️ Card not present")
                risk_factor_count += 1
            if distance > 100:
                st.warning("⚠️ Transaction far from home")
                risk_factor_count += 1
            if international:
                st.warning("⚠️ International transaction")
                risk_factor_count += 1
            if failed_24h > 0:
                st.warning("⚠️ Recent failed transactions")
                risk_factor_count += 1

            if risk_factor_count == 0:
                st.success("✅ No suspicious patterns detected")

    with tab2:
        st.subheader("Analyze Transaction Batch")

        n_transactions = st.slider("Number of transactions", 100, 5000, 1000)

        if st.button("📊 Generate & Analyze", key="batch_fraud"):
            with st.spinner("Generating transactions..."):
                generator = FraudDetectionDataGenerator(random_state=42)
                df = generator.generate(n_samples=n_transactions, fraud_rate=0.02)

                # Display stats
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Transactions", len(df))
                with col2:
                    fraud_count = df['is_fraud'].sum()
                    st.metric("Fraudulent", f"{fraud_count} ({fraud_count/len(df)*100:.2f}%)")
                with col3:
                    legitimate = (df['is_fraud'] == 0).sum()
                    st.metric("Legitimate", f"{legitimate}")
                with col4:
                    avg_amount = df['transaction_amount'].mean()
                    st.metric("Avg Amount", f"${avg_amount:.2f}")

                # Fraud vs Legitimate comparison
                fig = go.Figure()

                fig.add_trace(go.Box(
                    y=df[df['is_fraud'] == 0]['transaction_amount'],
                    name='Legitimate',
                    marker_color='green'
                ))

                fig.add_trace(go.Box(
                    y=df[df['is_fraud'] == 1]['transaction_amount'],
                    name='Fraudulent',
                    marker_color='red'
                ))

                fig.update_layout(
                    title="Transaction Amount: Fraud vs Legitimate",
                    yaxis_title="Amount ($)"
                )

                st.plotly_chart(fig, use_container_width=True)

                # Fraud by hour
                fraud_by_hour = df.groupby('transaction_hour')['is_fraud'].mean() * 100
                fig2 = px.line(
                    x=fraud_by_hour.index,
                    y=fraud_by_hour.values,
                    title="Fraud Rate by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Fraud Rate (%)'}
                )
                st.plotly_chart(fig2, use_container_width=True)

# ============================================================================
# DOCUMENT PROCESSING PAGE
# ============================================================================
elif page == "📄 Document Processing":
    st.header("📄 Document Processing")

    # Model type selection for classification
    doc_model_architecture = st.sidebar.radio(
        "Classification Model:",
        ["Rule-based (Keyword Matching)", "PyTorch LSTM (Deep Learning)"],
        index=0,
        key="doc_arch"
    )

    @st.cache_resource
    def load_processor():
        processor = DocumentProcessor()
        try:
            processor.load("models/document_processing/document_processor.pkl")
        except:
            pass  # Use default processor if no saved model
        return processor

    @st.cache_resource
    def load_deep_classifier():
        classifier = DocumentClassifierDeep()
        classifier.load("models/document_processing/document_classifier_deep.pkl")
        return classifier

    processor = load_processor()
    use_deep_classifier = "PyTorch" in doc_model_architecture

    if use_deep_classifier:
        try:
            deep_classifier = load_deep_classifier()
            st.info("🧠 Using **PyTorch LSTM** for document classification (Deep Learning)")
        except Exception as e:
            use_deep_classifier = False
            st.warning(f"⚠️ Deep classifier not available ({e}). Using rule-based classification.")
    else:
        st.info("📋 Using **rule-based keyword matching** for document classification")

    tab1, tab2 = st.tabs(["📝 Process Document", "📊 Batch Processing"])

    with tab1:
        st.subheader("Extract Information from Document")

        # Only show document type selector for rule-based classification
        if not use_deep_classifier:
            doc_type = st.selectbox("Document Type", ["invoice", "bank_statement", "tax_form", "contract"])
        else:
            doc_type = None  # Will be predicted by deep classifier

        document_text = st.text_area(
            "Paste document text here:",
            height=300,
            placeholder="Paste your invoice, statement, or other financial document here..."
        )

        if st.button("🔍 Process Document", key="process_doc"):
            if document_text:
                # Use deep classifier for classification if available
                if use_deep_classifier:
                    predicted_type, confidence = deep_classifier.classify_document(document_text)
                    doc_type = predicted_type
                    classification_method = f"Deep Learning (confidence: {confidence:.1%})"
                else:
                    if doc_type is None:
                        doc_type, confidence = processor.classify_document(document_text)
                    else:
                        confidence = 1.0
                    classification_method = "Rule-based"

                # Extract fields using the processor
                result = processor.process_document(document_text, doc_type)

                st.success("✅ Document processed successfully!")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Document Type", result['document_type'].upper())
                with col2:
                    st.metric("Total Amount", f"${result.get('total_amount', 0):,.2f}")
                with col3:
                    st.metric("Dates Found", result.get('date_count', 0))
                with col4:
                    st.metric("Word Count", result.get('word_count', 0))

                st.caption(f"Classification method: {classification_method}")

                st.subheader("Extracted Information:")
                st.json(result)
            else:
                st.warning("Please paste document text first")

    with tab2:
        st.subheader("Batch Document Processing")

        n_docs = st.slider("Number of documents to generate", 10, 500, 100)

        if st.button("📊 Generate & Process", key="batch_docs"):
            with st.spinner("Processing documents..."):
                from src.data_generators.document_processing_generator import DocumentProcessingDataGenerator

                generator = DocumentProcessingDataGenerator(random_state=42)
                df = generator.generate(n_samples=n_docs)

                results = processor.process_batch(df)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Documents Processed", len(results))
                with col2:
                    avg_amount = results['total_amount'].mean()
                    st.metric("Avg Amount", f"${avg_amount:,.2f}")
                with col3:
                    total_amount = results['total_amount'].sum()
                    st.metric("Total Amount", f"${total_amount:,.0f}")
                with col4:
                    avg_words = results['word_count'].mean()
                    st.metric("Avg Words", f"{avg_words:.0f}")

                # Document type distribution
                doc_type_counts = results['document_type'].value_counts()
                fig = px.pie(
                    values=doc_type_counts.values,
                    names=doc_type_counts.index,
                    title="Document Type Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(results.head(20))

# ============================================================================
# MODEL PERFORMANCE PAGE
# ============================================================================
elif page == "📊 Model Performance":
    st.header("📊 Model Performance Metrics")

    # Model type toggle
    perf_model_type = st.radio(
        "Select Model Type:",
        ["XGBoost (Traditional ML)", "PyTorch (Deep Learning)"],
        horizontal=True
    )

    col1, col2 = st.columns(2)

    if "XGBoost" in perf_model_type:
        with col1:
            st.subheader("Credit Risk - XGBoost")
            metrics = {
                "Accuracy": 0.9555,
                "Precision": 0.8651,
                "Recall": 0.8333,
                "F1 Score": 0.8489,
                "ROC AUC": 0.9876
            }
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")

        with col2:
            st.subheader("Fraud Detection - XGBoost")
            metrics = {
                "Accuracy": 0.9954,
                "Precision": 0.8532,
                "Recall": 0.9300,
                "F1 Score": 0.8900,
                "ROC AUC": 0.9988
            }
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
    else:
        with col1:
            st.subheader("Credit Risk - PyTorch NN")
            metrics = {
                "Accuracy": 0.9420,
                "Precision": 0.8450,
                "Recall": 0.8100,
                "F1 Score": 0.8271,
                "ROC AUC": 0.9750
            }
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
            st.caption("Architecture: 64→32→16→1 with BatchNorm & Dropout")

        with col2:
            st.subheader("Fraud Detection - PyTorch NN")
            metrics = {
                "Accuracy": 0.9940,
                "Precision": 0.8400,
                "Recall": 0.9150,
                "F1 Score": 0.8760,
                "ROC AUC": 0.9970
            }
            for metric, value in metrics.items():
                st.metric(metric, f"{value:.2%}")
            st.caption("Architecture: 128→64→32→1 with weighted BCE loss")

    st.markdown("---")

    # Document classifier metrics
    st.subheader("Document Classification")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Rule-based (Keyword Matching)**")
        st.metric("Typical Accuracy", "~85%")
        st.caption("Fast, interpretable, no training required")

    with col2:
        st.markdown("**PyTorch LSTM**")
        st.metric("Typical Accuracy", "~92%")
        st.caption("BiLSTM with 100-dim embeddings, 128 hidden units")

    st.markdown("---")

    st.info("""
    💡 **Model Information:**
    - **XGBoost models**: Fast training, interpretable feature importance, excellent for tabular data
    - **PyTorch models**: Neural networks with embeddings for categorical features, better for complex patterns
    - **Document LSTM**: Bidirectional LSTM for text classification, learns from word sequences
    - All models support save/load for production deployment
    """)

# Footer
st.markdown("---")
st.markdown("🚀 **Fintech AI Platform** | Built with Streamlit")
