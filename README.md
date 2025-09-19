# Synthetic Asthma — EDA & Classification

**Project summary:** EDA and binary classification on a synthetic asthma dataset (Kaggle). The notebook performs data cleaning, exploratory analysis, feature encoding, and trains two classifiers (KNN and Logistic Regression).

## Dataset
- **Source:** Kaggle (synthetic asthma patients dataset)
- **Filename (from notebook):** `synthetic_asthma_dataset.csv`
- **Total rows:** 10000
- **Columns (17):** see table below

| Column | Non-Null Count | Missing | Dtype |
|---|---:|---:|---|
| Patient_ID | 10000 | 0 | object |
| Age | 10000 | 0 | int64 |
| Gender | 10000 | 0 | object |
| BMI | 10000 | 0 | float64 |
| Smoking_Status | 10000 | 0 | object |
| Family_History | 10000 | 0 | int64 |
| Allergies | 7064 | 2936 | object |
| Air_Pollution_Level | 10000 | 0 | object |
| Physical_Activity_Level | 10000 | 0 | object |
| Occupation_Type | 10000 | 0 | object |
| Comorbidities | 5033 | 4967 | object |
| Medication_Adherence | 10000 | 0 | float64 |
| Number_of_ER_Visits | 10000 | 0 | int64 |
| Peak_Expiratory_Flow | 10000 | 0 | float64 |
| FeNO_Level | 10000 | 0 | float64 |
| Has_Asthma | 10000 | 0 | int64 |
| Asthma_Control_Level | 2433 | 7567 | object |

**Notes from data info:**
- `Allergies` has missing values (7064 non-null -> 2936 missing).
- `Comorbidities` has many missing values (5033 non-null -> 4967 missing).
- `Asthma_Control_Level` is largely missing (2433 non-null -> 7567 missing).

## Preprocessing & Cleaning (as implemented in the notebook)
- Filled missing values for columns like `Comorbidities`, `Asthma_Control_Level`, and `Allergies` using `fillna()` with appropriate defaults (e.g., `'None'`, `'Unknown'`, `'No Allergies'`).
- Dropped the `Patient_ID` column as an identifier that shouldn't be used for modeling.
- Created a modeling dataframe `df_model = df.drop('Asthma_Control_Level', axis=1)` and encoded categorical variables using `pd.get_dummies(df_model, dtype=int, drop_first=True)`.
- Split data into features and target: `x = df_model.drop('Has_Asthma', axis=1)`, `y = df_model['Has_Asthma']`.
- Train/test split with `test_size=0.2, random_state=42` (8000 train / 2000 test).

## Exploratory Data Analysis (EDA) - Key findings (notebook + your summary)
- There are more non-asthmatic patients than asthmatic patients (class imbalance).
- In the test set: **1504** non-asthmatic vs **496** asthmatic (≈75% vs 25%).
- Most asthmatic patients are not smokers; although, some are current smokers.
- Many asthmatic patients work indoors and report allergies; most do not report comorbidities.
- `Peak_Expiratory_Flow` rises and peaks around 400, then drops. `FeNO_Level` peaks around 25 and fades near 60. (See visualizations in the notebook.)
- Example plot included in the notebook: `ER Visits by Asthma Status` (boxplot).

## Modeling
Two classifiers were trained and evaluated on the dataset (features encoded with one-hot and no explicit scaling in the notebook):

- **K-Nearest Neighbors (KNN)**
  - Parameters used: `n_neighbors=5`
  - KNN was trained on the encoded features without feature scaling (consider scaling for better KNN performance).

- **Logistic Regression**
  - Parameters used: `max_iter=1000`
  - Trained and evaluated on the same train/test split.

## Model Evaluation & Results (from notebook outputs)
**Test set size:** 2000 (20% of 10000)

### Classification reports (as printed in the notebook)
```
precision    recall  f1-score   support

           0       0.75      0.92      0.83      1504
           1       0.24      0.08      0.12       496

    accuracy                           0.71      2000
   macro avg       0.49      0.50      0.47      2000
weighted avg       0.62      0.71      0.65      2000
```
```
precision    recall  f1-score   support

           0       0.98      1.00      0.99      1504
           1       1.00      0.95      0.97       496

    accuracy                           0.99      2000
   macro avg       0.99      0.97      0.98      2000
weighted avg       0.99      0.99      0.99      2000
```

### Confusion matrices (as printed in the notebook)
```
Confusion Matrix:
[[1381  123]
 [ 458   38]]
```
```
Confusion Matrix:
[[1504    0]
 [  25  471]]
```

**Key reported accuracies (from notebook):**
- Accuracy: **0.71**
- Accuracy: **0.99**

## Conclusions & Next Steps
- Logistic Regression shows very high accuracy (~0.99). This is excellent but should be checked for potential data leakage or class imbalance effects.
- KNN scored ~0.71 — try scaling features (StandardScaler) and hyperparameter tuning (`GridSearchCV` on `n_neighbors`) to improve KNN performance.
- Consider stratified sampling, cross-validation, and additional metrics (ROC-AUC) due to class imbalance.
- Because the dataset is synthetic, validate model behavior on real-world data before any deployment or clinical inference.

## How to run the notebook
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```
2. Start Jupyter and open `Synthetic_Asthma EDA.ipynb` and run cells in order.

## Files
- `Synthetic_Asthma EDA.ipynb` — primary notebook (contains all code, outputs, and visualizations).
- `README.md` — this file.
