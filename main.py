from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report, confusion_matrix

from src.preprocessing import encode_data
from src.data_cleaning import data_clean_and_handling_missing_values
from src.feature_engineering import feature_engineering 
from src.train import train_model
from src.evaluate import evaluate_model
import pandas as pd

# Load dataset

df = pd.read_csv("data/german_credit_data.csv")

#Data Cleaning and Handling Missing Values
df = data_clean_and_handling_missing_values(df)

#Feature Engineering
df = feature_engineering(df)

#Splitting data into features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

#Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#One Hot Encoding for categorical variables
X_train_final, X_test_final = encode_data(X_train, X_test)

#Model Training
model = train_model(X_train_final, y_train)

#Model Evaluation
evaluate_model(model, X_test_final, y_test)

