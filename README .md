# 🔍 ChurnRadar — Customer Churn Prediction App

A Streamlit-based web application that predicts customer churn probability using a Logistic Regression model. The app features an interactive dashboard with real-time predictions, feature importance analysis, and what-if scenario simulations.

---

## 📸 Overview

ChurnRadar lets you input customer profile data and instantly see:
- Churn probability score with risk classification
- Model performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Feature importance and coefficient analysis
- Score distribution relative to the full customer base
- What-if scenario comparisons for retention planning

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
git clone https://github.com/your-username/churnradar.git
cd churnradar
pip install -r requirements.txt
```

### Required Files

Before running the app, make sure the following model artifact files are present in the project root:

| File | Description |
|------|-------------|
| `Customer_churn_logistic_model.joblib` | Trained Logistic Regression model |
| `Customer_churn_scaler.joblib` | Fitted StandardScaler |
| `Customer_churn_encoders.joblib` | Fitted LabelEncoders for categorical features |
| `Customer_churn_feature_names.joblib` | Ordered list of feature names |
| `metrics.json` | Model evaluation metrics (accuracy, precision, recall, F1, ROC-AUC, coefficients) |

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| Algorithm | Logistic Regression |
| Encoding | LabelEncoder (categorical features) |
| Scaling | StandardScaler |
| Training set size | ~2,000 customers |
| Number of features | 11 |

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Customer age (18–80) |
| `Tenure` | Numeric | Months as a customer (1–72) |
| `MonthlyCharges` | Numeric | Monthly bill amount ($20–$150) |
| `TotalCharges` | Numeric | Cumulative charges ($0–$15,000) |
| `CustomerServiceCalls` | Numeric | Number of service calls made (0–15) |
| `DataUsageGB` | Numeric | Average monthly data usage in GB (0–500) |
| `ContractType` | Categorical | Month-to-month / One year / Two year |
| `PaymentMethod` | Categorical | Bank transfer / Credit card / Electronic check / Mailed check |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `TechSupport` | Categorical | Yes / No |
| `OnlineSecurity` | Categorical | Yes / No |

---

## 📊 Dashboard Features

### Left Panel — Customer Inputs
Fill in customer details across three sections:
- **Account Info** — Age, Tenure, Service Calls, Data Usage
- **Billing** — Monthly and Total Charges
- **Plan Details** — Contract type, Payment method, Internet service, Tech support, Online security

Click **⚡ Run Prediction** to generate results.

### Right Panel — Results & Analysis

**Model KPIs** — Live metrics bar at the top showing Accuracy, Precision, Recall, F1 Score, and ROC-AUC.

**Prediction Hero Card** — Large result display showing:
- Churn / No-Churn verdict
- Exact probability percentage
- Risk level (Low / Medium / High)
- Segmented risk bar

**Gauge Chart** — Needle-style gauge showing the churn probability score.

**Risk Factors** — Color-coded pills highlighting the top positive and negative churn drivers for the current customer.

**Feature Sensitivity** — Bar chart showing how sensitive the prediction is to each input feature.

**Analysis Tabs:**

| Tab | Content |
|-----|---------|
| Confusion Matrix | Model confusion matrix heatmap |
| Feature Importance | Signed LR coefficients + absolute importance ranking |
| Score Distribution | Histogram of churn vs non-churn scores with current customer highlighted |

**What-If Scenarios** — Side-by-side comparison of churn probability under four conditions:
1. Current profile
2. Upgrading to a 2-year contract
3. Adding Tech Support + Online Security
4. Reducing monthly bill by 20%

**Retention Recommendation** — Automatically highlights the best intervention to reduce churn risk.

---

## 📦 Dependencies

```
streamlit
joblib
numpy
pandas
plotly
scikit-learn
```

Install all dependencies with:

```bash
pip install streamlit joblib numpy pandas plotly scikit-learn
```

---

## 📁 Project Structure

```
churnradar/
├── app.py                                    # Main Streamlit application
├── Customer_churn_logistic_model.joblib      # Trained model
├── Customer_churn_scaler.joblib              # Feature scaler
├── Customer_churn_encoders.joblib            # Categorical encoders
├── Customer_churn_feature_names.joblib       # Feature name list
├── metrics.json                              # Evaluation metrics & coefficients
├── requirements.txt                          # Python dependencies
└── README.md                                 # This file
```

---

## ⚠️ Notes

- The score distribution chart in Tab 3 uses synthetic data (`np.random.seed(42)`) to illustrate the population distribution. It is for visual reference only and does not reflect the actual training dataset.
- Feature importance is derived from the absolute values of Logistic Regression coefficients, not from a tree-based `feature_importances_` attribute.
- The app requires all five model artifact files to be present at startup — it will fail to load otherwise.
