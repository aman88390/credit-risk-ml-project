# Credit Risk Prediction using Machine Learning

## 📌 Problem Statement

The objective of this project is to predict whether a customer is **risky (bad)** or **safe (good)** for granting a loan based on their financial and demographic information.

This is a **binary classification problem** where:

* **0 → Bad (risky customer)**
* **1 → Good (safe customer)**

---

## 🎯 Project Objective

* Build a machine learning model to identify risky customers
* Improve detection of bad loans using appropriate techniques
* Develop a **clean, modular pipeline** for real-world usability

---

## 📊 Dataset Overview

The dataset contains customer-related information such as:

* Credit amount
* Duration
* Age
* Job
* Housing
* Saving accounts
* Checking account
* Purpose

The dataset is **imbalanced**, with more good customers than bad ones.

---

## ⚙️ Approach

### 1. Data Cleaning

* Handled missing values in:

  * Saving accounts
  * Checking account
* Missing values treated as **"Unknown"** category

---

### 2. Exploratory Data Analysis (EDA)

Key insights:

* Dataset is imbalanced (~70% good, ~30% bad)
* Higher credit amount → higher risk
* Longer duration → higher risk
* Low savings/checking balance → higher risk

---

### 3. Feature Engineering

Created meaningful features:

* **Credit_per_month** → captures repayment burden
* **Saving_group** → low / medium / high
* **Checking_group** → low / medium / high

---

### 4. Encoding

* Applied **One-Hot Encoding using `OneHotEncoder`**
* Used:

  * `drop='first'` → avoid redundancy
  * `handle_unknown='ignore'` → handle unseen categories

---

### 5. Model Training

* Used **Logistic Regression** as baseline model
* Chosen for:

  * Simplicity
  * Interpretability
  * Stable performance

---

### 6. Handling Class Imbalance

* Applied **SMOTE (Synthetic Minority Oversampling Technique)**
* Improved model performance on minority class (bad customers)

---

## 📈 Results

| Metric             | Before SMOTE | After SMOTE    |
| ------------------ | ------------ | -------------- |
| Recall (Bad Class) | Lower        | Improved       |
| Accuracy           | Higher       | Slightly lower |

### 🔑 Key Insight

In credit risk problems:

> **Recall is more important than accuracy**
> Because missing a bad customer can lead to financial loss.

---

## 🧠 Final Outcome

* Model is better at identifying risky customers
* Balanced performance across classes
* Suitable for real-world credit risk assessment

---

## 📁 Project Structure

```
CREDIT_RISK_PROJECT/
│
├── data/
│   └── german_credit_data.csv
│
├── notebooks/
│   ├── Credit_Risk_Final.ipynb
│   └── eda_experiment.ipynb
│
├── src/
│   ├── data_cleaning.py
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── train.py
│   └── models/
│       └── credit_risk_model.pkl
│
├── venv/
├── .gitignore
├── main.py
├── readme.md
└── requirements.txt
```

---

## 🚀 How to Run the Project

### 1. Clone the repository

```
git clone <your-repo-link>
cd credit-risk-project
```

### 2. Install dependencies

```

pip install -r requirements.txt
```

### 3. Run the pipeline

```
python main.py
```

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Imbalanced-learn (SMOTE)
* Matplotlib / Seaborn

---

## 🔍 Key Learnings

* Importance of handling class imbalance
* Feature engineering improves model performance
* Proper pipeline structure is critical for real-world ML
* Trade-off between accuracy and recall

---

## 🔮 Future Improvements

* Use advanced models (XGBoost, LightGBM)
* Hyperparameter tuning
* Cross-validation
* Deploy as API (Flask/FastAPI)
* Build UI using Streamlit

---

## 👤 Author

Aman Yadav

---

## ⭐ Final Note

This project demonstrates a complete **end-to-end machine learning pipeline**, from data analysis to model deployment-ready structure.
