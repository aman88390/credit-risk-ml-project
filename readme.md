# Credit Risk Prediction (Machine Learning Project)

## 📌 Problem Statement

The objective of this project is to predict whether a customer is likely to default on a loan based on their financial and demographic details.

---

## 📊 Dataset

* German Credit Dataset
* Contains customer information such as:

  * Age
  * Sex
  * Job
  * Housing
  * Saving accounts
  * Checking account
  * Credit amount
  * Duration
  * Purpose
* Target variable:

  * **Risk**

    * good → 0 (no default)
    * bad → 1 (default)

---

## 🚀 Current Progress

### ✔ Data Loading

* Loaded dataset using pandas
* Verified data using `df.head()`

### ✔ Data Understanding

* Checked structure using `df.info()`
* Analyzed numerical features using `df.describe()`
* Analyzed categorical features using `df.describe(include="object")`

---

## 🔍 Initial Observations

* Dataset contains both numerical and categorical features
* Target variable is categorical (good/bad)
* Missing values present in:

  * Saving accounts
  * Checking account
* Credit amount and loan duration vary significantly across customers

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn (planned)

---

## 📂 Project Structure


credit_risk_project/
│
├── data/
│   └── german_credit_data.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── README.md

---

## 📈 Next Steps

* Data Cleaning
* Feature Engineering
* Exploratory Data Analysis (visual)
* Model Training and Evaluation
* Deployment (Flask API)
