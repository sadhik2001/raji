import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.preprocess import load_and_preprocess_data

def evaluate_model(model_path='models/random_forest.pkl', csv_path='data/creditcard.csv'):
    # Load test data
    _, X_test, _, y_test = load_and_preprocess_data(csv_path)
    
    # Load model
    model = joblib.load(model_path)
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # For ROC AUC
    
    # Evaluation metrics
    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n✅ Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\n✅ ROC AUC Score:")
    print(roc_auc_score(y_test, y_proba))

if __name__ == "__main__":
    evaluate_model()
