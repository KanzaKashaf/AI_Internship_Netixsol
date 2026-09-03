# README — Reproducing Week 1 Day 4 Results

## Project

**Adult Census Income Classification — Model Tuning, Calibration and Reproducible Pipelines**

This notebook continues the Adult Census Income classification project from the previous tasks. The objective is to predict whether an individual's annual income is greater than $50K.

The primary business objective is to maximize **precision** in order to reduce false-positive predictions and avoid wasted outreach.

---

## Dataset

The dataset is loaded directly using:

```python
from sklearn.datasets import fetch_openml

adult = fetch_openml(
    "adult",
    version=2,
    as_frame=True
)
```

The dataset is automatically downloaded from OpenML and cached locally by scikit-learn.

---

## Required Libraries

Install the required libraries before running the notebook:

```bash
pip install numpy pandas scikit-learn scipy matplotlib seaborn joblib
```

---

## Library Versions

The experiments were executed using the following environment:

* Python: **3.14.6**
* NumPy: **2.4.6**
* Pandas: **3.0.3**
* Scikit-learn: **1.9.0**
* SciPy: **1.18.0**

Using different library versions may result in small differences in model performance or hyperparameter search results.

---

## Reproducibility

A fixed random state is used throughout the notebook:

```python
RANDOM_STATE = 42
```

Randomness is controlled in:

* Train/test splitting
* Stratified cross-validation
* RandomizedSearchCV
* Logistic Regression
* HistGradientBoostingClassifier
* Other operations where randomness is applicable

The dataset is split using:

```python
test_size = 0.20
stratify = target
random_state = 42
```

This produces a reproducible hold-out test set.

---

## Workflow

Run the notebook cells sequentially from top to bottom.

The complete workflow is:

1. Load the Adult Census Income dataset.
2. Replace missing-value markers with NaN.
3. Convert the income target into binary form.
4. Recreate the reproducible 80/20 stratified train/test split.
5. Apply feature engineering.
6. Apply numeric and categorical preprocessing.
7. Build reproducible machine learning pipelines.
8. Tune Logistic Regression using RandomizedSearchCV.
9. Tune HistGradientBoosting using RandomizedSearchCV.
10. Compare models using 5-fold Stratified Cross-Validation.
11. Analyze learning curves and regularization behavior.
12. Evaluate probability calibration.
13. Select an appropriate classification threshold.
14. Perform final evaluation on the hold-out test set.
15. Save the final model artifact.

---

## Feature Engineering

The following engineered features are created:

* `age_bucket`
* `hours_bucket`
* `capital_gain_flag`
* `log_capital_gain`
* `higher_education`
* `education_hours_interaction`

All feature engineering uses information from the current row only and does not use the target variable.

---

## Preprocessing

### Numeric Features

The numeric preprocessing pipeline applies:

1. Median imputation
2. Standard scaling

### Categorical Features

The categorical preprocessing pipeline applies:

1. Most-frequent imputation
2. One-hot encoding

The setting:

```python
handle_unknown="ignore"
```

ensures that unseen categories do not cause errors during inference.

---

## Hyperparameter Optimization

Hyperparameter tuning uses:

```python
StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
```

The primary optimization metric is:

```python
precision
```

The tuned models are:

* Logistic Regression
* HistGradientBoostingClassifier

---

## Final Model

The final selected model is a calibrated HistGradientBoosting pipeline.

The final operating threshold is:

```python
FINAL_THRESHOLD = 0.80
```

The threshold is intentionally higher than the default threshold of 0.50 because the business objective prioritizes precision and minimizing false positives.

---

## Final Model Artifact

The trained model is saved using Joblib:

```python
MODEL_PATH = "adult_income_final_pipeline.joblib"

joblib.dump(
    final_model,
    MODEL_PATH
)
```

The artifact can be loaded using:

```python
model = joblib.load(
    "adult_income_final_pipeline.joblib"
)
```

---

## How to Make Predictions

Provide new data in the same raw feature format as the original Adult dataset.

Example:

```python
new_data = X_test.iloc[[0]].copy()

probability = model.predict_proba(
    new_data
)[:, 1]

prediction = (
    probability >= 0.80
).astype(int)
```

The pipeline automatically performs:

* Feature engineering
* Missing-value handling
* Numeric scaling
* Categorical encoding
* Probability prediction

No manual preprocessing is required before inference.

---

## Important Evaluation Rule

The hold-out test set should not be used during repeated model development or hyperparameter tuning.

The recommended workflow is:

```text
Training Data
      ↓
Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
Model Selection
      ↓
Calibration and Threshold Selection
      ↓
Final Model Locked
      ↓
Hold-Out Test Evaluation
```

This helps provide an unbiased estimate of final model performance.

---

## Production Considerations

In production, the model should be monitored for:

* Changes in feature distributions
* New or unseen categories
* Missing-value patterns
* Prediction distribution changes
* Precision and recall degradation
* Changes in the business cost of false positives and false negatives

The selected threshold of 0.80 should also be periodically reviewed because the optimal operating threshold depends on business requirements and the relative cost of different prediction errors.
