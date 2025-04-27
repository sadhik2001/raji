import joblib
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train, model_path='models/random_forest.pkl'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
    
    return model
