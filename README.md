
<h1>📉 Customer Churn Prediction</h1>  

This project is focused on predicting customer churn for a telecom company using machine learning techniques. It includes data preprocessing, model training, evaluation, and deployment using `scikit-learn`, `XGBoost`, and `SMOTE` for imbalanced classification.

---

## 🧰 Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn (for visualization)
* Scikit-learn
* XGBoost
* imbalanced-learn (SMOTE)
* Pickle (for saving models and encoders)

---

## 📁 Dataset

Dataset used: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
It contains 7043 customer records with 21 attributes including:

* Customer demographics
* Account information
* Services signed up for
* Monthly and total charges
* Target variable: `Churn` (Yes/No)

---

## 🧪 Project Steps

### 1. Importing Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
```

---

### 2. Data Loading & Preprocessing

* Read the CSV data into a DataFrame.
* Perform label encoding for all categorical features.
* Save the encoders using `pickle` for future inference.

---

### 3. Train-Test Split

* `X`: All features except `Churn`
* `y`: Target variable `Churn`
* 80% training / 20% test split

---

### 4. Handling Imbalanced Data with SMOTE

* Original training set is imbalanced:

  ```
  0 (No churn): 4138
  1 (Churn):    1496
  ```
* Applied **SMOTE** to balance the classes:

  ```
  0: 4138
  1: 4138
  ```

---

### 5. Model Training & Cross Validation

Trained three classifiers using 5-fold cross-validation:

| Model         | Accuracy (avg) |
| ------------- | -------------- |
| Decision Tree | 0.78           |
| Random Forest | 0.84 ✅         |
| XGBoost       | 0.83           |

✅ **Random Forest** performed the best.

---

### 6. Model Evaluation on Test Set

* Accuracy: **77.85%**
* Precision (Churn class): **58%**
* Confusion Matrix:

```
[[878 158]
 [154 219]]
```

---

### 7. Saving the Model

* Saved the `RandomForestClassifier` and feature names using `pickle` as `customer_churn_model.pkl`.
* Encoders saved as `encoders.pkl`.

---

### 8. Predictive System

Created a sample input and performed the following:

* Loaded saved model and encoders.
* Transformed categorical fields using `LabelEncoder`.
* Generated churn prediction and probability.

**Example Output**:

```
Prediction: No Churn
Prediction Probability: [[0.78 0.22]]
```

---

## ✅ To Do

* [ ] Implement Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
* [ ] Model Selection using metrics beyond accuracy (ROC-AUC, F1-score)
* [ ] Try Downsampling (RandomUnderSampler)
* [ ] Address Overfitting (feature reduction, regularization)
* [ ] Use StratifiedKFold Cross Validation for consistent sampling
---

## 🧠 Conclusion

This churn prediction model can help telecom providers proactively identify at-risk customers and take retention actions. With further enhancements such as hyperparameter tuning and advanced sampling strategies, its performance can be further improved.
