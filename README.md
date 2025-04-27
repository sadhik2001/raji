Built for the GrowthLink Data Science Internship Program.

# credit-card-fraud-detection:- 
A machine learning project to detect fraudulent credit card transactions using the popular Kaggle dataset.

## 📁 credit-card-fraud-detection- Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv               # Dataset (downloaded from Kaggle)
│
├── notebooks/
│   └── eda.ipynb                    # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py                # Preprocessing steps
│   ├── train_model.py               # Model training
│   ├── evaluate.py                  # Model evaluation
│
├── main.py                          # Run the whole project
├── requirements.txt                 # Dependencies
├── README.md                        # GitHub ReadMe
└── .gitignore                       # Ignore unnecessary files

```

---

## 📊 Dataset

- Download it from: [Kaggle Dataset Link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Contains 284,807 transactions with 492 frauds (highly imbalanced)

---

## ⚙️ How to Run the Project

```bash
# Step 1: Clone the repo or download project
```bash
   git clone https://github.com/sadhik2001/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Make sure dataset is in data/creditcard.csv

# Step 4: Run the main script or Train the model:
python main.py

# Step 5: Evaluate the model:
python src/evaluate.py


```

---

## 🧪 Output Example

When you run it, you’ll see output like:

```
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     56863
           1       0.91      0.84      0.87        99

    accuracy                           1.00     56962
   macro avg       0.96      0.92      0.94     56962
weighted avg       1.00      1.00      1.00     56962
```

And a confusion matrix plot with actual vs predicted labels.

---

📈 Model
Uses a Random Forest classifier with SMOTE for handling class imbalance.

📊 Metrics
Example results:

Accuracy: 99.8%

Precision: 0.94

Recall: 0.89

ROC AUC: 0.97

## 📌 What we Did

✅ Data cleaning & scaling  
✅ Handling imbalance using SMOTE  
✅ Building models using sklearn  
✅ Evaluating performance with F1, precision, recall

---



