from src.preprocess import load_and_preprocess_data
from src.train_model import train_model
from sklearn.metrics import classification_report, confusion_matrix

def main():
    print("🚀 Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data/creditcard.csv")
    
    print("🎯 Training model...")
    model = train_model(X_train, y_train)
    
    print("📊 Evaluating model...")
    y_pred = model.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()

