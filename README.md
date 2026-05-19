# Loan Prediction 

## 📌 Project Overview

This project is an end-to-end Machine Learning classification project built using Python and Scikit-learn.  
The objective of this project is to predict whether a loan application will be approved or not using a Multivariate Logistic Regression model.

The project includes:
- Data preprocessing
- Missing value handling
- Feature scaling
- Feature selection
- Logistic Regression model training
- Model evaluation
- SHAP explainability

---

## 📂 Dataset Information

**Dataset:** Loan Prediction Problem Dataset

**Source:**  
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

The dataset contains applicant information such as:
- Gender
- Marital Status
- Education
- Applicant Income
- Loan Amount
- Credit History
- Property Area
- Loan Status

### Target Variable

- `Loan_Status`
  - `Y` → Loan Approved
  - `N` → Loan Rejected

---

## 🚀 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP
- KaggleHub

---

## ⚙️ Project Workflow

1. Load Dataset
2. Perform Exploratory Data Analysis (EDA)
3. Handle Missing Values
4. Encode Categorical Features
5. Scale Numerical Features
6. Perform Train-Test Split
7. Perform Feature Selection
8. Train Logistic Regression Model
9. Evaluate Model Performance
10. Generate SHAP Explainability Plots

---

## 🧹 Data Preprocessing

- Missing values handled using `SimpleImputer()`
- Numerical features scaled using `StandardScaler()`
- Categorical features encoded using `OneHotEncoder()`
- Preprocessing performed using `Pipeline` and `ColumnTransformer`

---

## 📊 Feature Selection

Feature selection was performed using:

```python
SelectKBest(score_func=chi2, k=5)
```

Top 5 important features were selected for model training.

---

## 🤖 Model Used

### Multivariate Logistic Regression

```python
LogisticRegression(max_iter=1000)
```

The model predicts whether a loan application is approved or rejected.

---

## 📈 Model Evaluation

The following evaluation metrics were used:

- Accuracy Score
- Confusion Matrix
- Classification Report
  - Precision
  - Recall
  - F1-Score

---

## 🧠 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to explain model predictions and feature importance.

Implemented using:

```python
shap.LinearExplainer()
```

SHAP Summary Plot helps visualize:
- Important features
- Positive impact on prediction
- Negative impact on prediction

---

## ▶️ How to Run the Project

### 1. Clone Repository

```bash
git clone https://github.com/your-username/loan-prediction-logistic-regression.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Python Script

```bash
python loan_prediction.py
```

---

## 📌 Output Files

The project generates:
- Confusion Matrix Plot
- Feature Importance Plot
- SHAP Summary Plot
- Model Evaluation Results

---

