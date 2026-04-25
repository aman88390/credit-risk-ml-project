# Model Training Script
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

def train_model(X_train, y_train):
    #Oversampling using SMOTE
    oversampler = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)
    
    #model training after resampling 
    model = LogisticRegression(max_iter=1000)
    
    joblib.dump(model, "src/models/credit_risk_model.pkl")
    model.fit(X_train_resampled, y_train_resampled)
    
    return model
    
    