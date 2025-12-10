# **Breast Cancer Detection – Machine Learning Mini Project**

### **CSC 334 Final Project – Alina Yildirim**

---

## **Problem Overview**

Breast cancer is one of the most common cancers worldwide, and early detection dramatically improves patient outcomes.
The goal of this project is to build models that can **predict whether a breast tumor is malignant or benign** using the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset.

This project demonstrates a clean, interpretable machine learning workflow suitable for biomedical applications.

---

## **📁 Project Structure & Approach**

### **1. Exploratory Data Analysis (EDA)**

* Class balance visualization
* Feature correlation heatmap
* Inspection of feature distributions

### **2. Data Preprocessing**

* Train/test split (80/20) with stratification
* Standard scaling for models that require normalization (Logistic Regression)

### **3. Model Training**

Two supervised learning models were trained:

1. **Logistic Regression**

   * Interpretable linear model
   * Trained on standardized features
   * Evaluated with accuracy, classification report, ROC curve

2. **Random Forest Classifier**

   * Ensemble-based nonlinear model
   * Handles high-dimensional interactions
   * Evaluated with confusion matrix, AUC, feature importance

### **4. Model Evaluation**

Included:

* Accuracy scores
* Confusion matrices for both models
* ROC curves + AUC
* Feature importance plots
* Misclassification patterns

These metrics help evaluate performance and **understand how models behave in a clinical context**.

### **5. Interpretation**

* Logistic Regression coefficients highlight which features push predictions toward malignant vs. benign.
* Random Forest importance identifies the most informative features (e.g., worst perimeter, mean concave points).

---

## **Models Used**

| Model                        | Why it was chosen                                                                   |
| ---------------------------- | ----------------------------------------------------------------------------------- |
| **Logistic Regression**      | Highly interpretable, fast, strong baseline for biomedical classification           |
| **Random Forest Classifier** | Handles nonlinear relationships, robust to noise, often high accuracy in medical ML |

---

## **Minimal Reproducible Example**

Below is a minimal script showing how to load the data, train a logistic regression model, and make predictions:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predict
preds = model.predict(X_test_scaled)
print("Test Accuracy:", accuracy_score(y_test, preds))
```

### **Expected Output**

```
Test Accuracy: ~0.97–0.99
```



## **Summary**

This project demonstrates:

* How to build and interpret ML models for biomedical classification
* The importance of evaluation metrics like ROC/AUC
* How model interpretability matters in medical settings

Both logistic regression and random forest perform extremely well, achieving high AUC and accuracy, and both can be valuable tools in supporting early cancer screening workflows.

