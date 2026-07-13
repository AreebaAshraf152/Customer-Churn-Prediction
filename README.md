# 📊 Telco Customer Churn Prediction

# Telco Customer Churn Prediction

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-xkjdk7wb8kvypnbervwegz.streamlit.app/)
> An end-to-end Machine Learning project that predicts customer churn for a telecommunications company using classification models and an interactive Streamlit web application.

---

## 📖 Project Overview

Customer retention is one of the most important business challenges in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one, making churn prediction a valuable business problem.

In this project, I developed a complete machine learning pipeline to predict whether a customer is likely to leave the company based on their demographic information, subscribed services, account details, and billing history.

Rather than focusing only on model accuracy, this project follows a real-world machine learning workflow—from data preprocessing and exploratory analysis to model comparison, hyperparameter tuning, explainability, and deployment through a Streamlit application.

---

## 🎯 Objectives

* Build an end-to-end customer churn prediction system.
* Perform comprehensive data cleaning and preprocessing.
* Explore customer behavior through Exploratory Data Analysis (EDA).
* Compare multiple machine learning algorithms.
* Optimize the best-performing model using hyperparameter tuning.
* Deploy the final model using Streamlit.
* Explain model decisions using feature importance.

---

# Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains information about telecom customers including:

* Customer demographics
* Subscription details
* Internet services
* Billing information
* Contract information
* Payment methods
* Customer churn status (Target Variable)

**Total Records:** 7,043

**Target Variable:** Churn (Yes / No)

---

# Project Workflow

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
One-Hot Encoding
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Feature Importance Analysis
      │
      ▼
Model Serialization
      │
      ▼
Streamlit Deployment
```

---

# Data Preprocessing

The preprocessing pipeline included:

* Handling missing values
* Converting data types
* Converting `TotalCharges` to numeric format
* One-Hot Encoding categorical variables
* Feature Scaling using **StandardScaler**
* Train-Test Split
* Saving the trained scaler for deployment

---

# Exploratory Data Analysis

Several visualizations were created to better understand customer behavior and churn patterns.

Key findings included:

* Customers with shorter tenure were more likely to churn.
* Month-to-month contracts had considerably higher churn rates.
* Fiber optic customers showed higher churn compared to DSL users.
* Customers paying through Electronic Check exhibited higher churn.
* Long-term contracts were associated with better customer retention.
* Monthly charges influenced churn probability more than total charges.

EDA included:

* Churn distribution
* Contract analysis
* Internet service analysis
* Correlation heatmap
* Feature relationship analysis
* Numerical feature distributions

---

# Feature Engineering

Categorical variables were transformed using **One-Hot Encoding**.

The final dataset contained **30 engineered features**, including:

* Contract type
* Internet service
* Payment method
* Streaming services
* Online security
* Technical support
* Device protection
* Paperless billing
* Customer demographics

---

# Machine Learning Models

Four different classification algorithms were trained and evaluated.

| Model                   |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression     |     80.62% |     65.93% |     55.88% |     60.49% |     84.22% |
| Decision Tree           |     72.53% |     48.25% |     47.86% |     48.05% |     64.60% |
| Random Forest           |     78.50% |     61.87% |     49.47% |     54.98% |     82.48% |
| **Tuned Random Forest** | **80.70%** | **67.47%** | **52.67%** | **59.16%** | **84.26%** |

The Tuned Random Forest model achieved the best overall balance between predictive performance and generalization, and was selected for deployment.

---

# Hyperparameter Tuning

To improve model performance, GridSearchCV was used for hyperparameter optimization.

The tuning process explored different combinations of:

* Number of trees
* Maximum tree depth
* Minimum samples split
* Minimum samples per leaf

The final optimized Random Forest model was saved using Pickle for deployment.

---

# Model Explainability

To improve model transparency, feature importance analysis was performed using the trained Random Forest model.

The Streamlit application includes:

* Sorted Feature Importance Chart
* Top 15 Most Important Features Table

This helps explain which customer attributes contribute most to churn predictions.

---

# Streamlit Web Application

A complete Streamlit application was developed to make predictions through an interactive user interface.
### 🚀 Live Demo

Try the deployed application here:

🔗 Streamlit App:
https://customer-churn-prediction-xkjdk7wb8kvypnbervwegz.streamlit.app/

### Features

* Customer information form
* Service information section
* Account information section
* Billing information section
* Real-time churn prediction
* Churn probability
* Model performance sidebar
* Feature importance visualization
* Top feature ranking table

---

# Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Streamlit
* Pickle
* VS Code
* Git & GitHub

---

# Project Structure

```
Telco-Customer-Churn-Prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── models/
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
│
├── images/
│
└── notebooks/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Telco-Customer-Churn-Prediction.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

# Future Improvements

* Deploy the application on Streamlit Community Cloud
* Add SHAP-based explainability
* Store prediction history
* Improve UI/UX with custom themes
* Add downloadable prediction reports
* Integrate a database for persistent storage

---

# Author

**Areeba Ashraf**

BS Data Science

Government College University Faisalabad

---

If you found this project interesting or have suggestions for improvement, feel free to connect or open an issue.
