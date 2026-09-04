# Adult Income Prediction — Machine Learning Project

## Project Objective

This project develops an end-to-end machine learning system to predict whether an individual's annual income exceeds $50K based on demographic, educational, employment, and financial attributes.

The selected business objective prioritizes **precision** to reduce the number of false-positive predictions in scenarios such as targeted marketing.

---

## Dataset

The project uses the **UCI Adult Census Income dataset**, accessed through OpenML using `fetch_openml`.

The dataset contains demographic and employment-related attributes such as:

* Age
* Workclass
* Education
* Education number
* Occupation
* Marital status
* Capital gain
* Capital loss
* Hours per week
* Native country

---

## Target Variable

The target variable is:

```text
0 → Income ≤ $50K
1 → Income > $50K
```

---

## Feature Engineering

The following engineered features were created:

1. Age buckets
2. Hours-per-week buckets
3. Capital gain indicator
4. Log-transformed capital gain
5. Higher education indicator
6. Education × hours-per-week interaction

All engineered features use only information available within the current observation.

---

## Preprocessing

### Numeric Features

* Median imputation
* Standard scaling

### Categorical Features

* Most-frequent imputation
* One-hot encoding
* Unknown categories ignored during inference

All preprocessing steps are included inside the machine learning pipeline.

---

## Models Tested

The project evaluated the following models:

* Majority-class baseline
* Rule-based baseline
* Logistic Regression
* Random Forest Classifier
* HistGradientBoostingClassifier

---

## Model Development

The project followed this workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Baseline Models
      ↓
Preprocessing Pipeline
      ↓
Supervised Model Comparison
      ↓
Feature Engineering
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Calibration
      ↓
Threshold Selection
      ↓
Final Validation
      ↓
Production Inference
```

---

## Hyperparameter Tuning

Hyperparameter optimization was performed using:

* `RandomizedSearchCV`
* 5-fold Stratified Cross-Validation
* Precision as the optimization metric

The primary model selected for tuning was `HistGradientBoostingClassifier`.

---

## Best Model

The selected final model is:

**Calibrated HistGradientBoostingClassifier**

The model was selected because it achieved the strongest cross-validated precision among the tested supervised models.

---

## Classification Threshold

The default classification threshold of `0.50` was adjusted.

The final selected threshold is:

**`0.80`**

The higher threshold prioritizes precision and reduces false-positive predictions.

---

## Final Test Performance

After running Day 5, replace this section with the verified results:

```text
Accuracy:
Precision:
Recall:
F1 Score:
ROC AUC:
PR AUC:
Brier Score:
```

---

## Model Interpretation

Permutation feature importance is used to interpret the final model.

The importance analysis measures how much the primary evaluation metric changes when the values of a feature are randomly shuffled.

Features that produce the largest performance decrease when shuffled are considered the most influential for the model.

---

## Production Inference

The final pipeline is saved as:

```text
final_model.joblib
```

The pipeline automatically performs:

* Feature engineering
* Missing-value handling
* Numeric scaling
* Categorical encoding
* Probability prediction

New data should be provided in the same raw format as the original dataset.

---

## Running Inference

```python
import joblib

model = joblib.load(
    "final_model.joblib"
)

probabilities = model.predict_proba(
    new_data
)[:, 1]

predictions = (
    probabilities >= 0.80
).astype(int)
```

No manual preprocessing is required.
