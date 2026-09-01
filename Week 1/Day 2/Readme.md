# Week 1 - Day 2: Baseline Machine Learning Models

## Overview

This project builds and evaluates baseline machine learning models for predicting whether an individual's annual income is **greater than $50K** using the Adult Income dataset.

Day 2 focuses on establishing a complete preprocessing and modeling pipeline, comparing baseline rules with Logistic Regression and Decision Tree models, and analyzing model errors and feature importance.

---

## Objectives

* Prepare numerical and categorical features using a reproducible preprocessing pipeline.
* Handle missing values and categorical variables.
* Train Logistic Regression and Decision Tree models.
* Evaluate models using multiple classification metrics.
* Compare models against simple baseline approaches.
* Analyze the confusion matrices and prediction errors.
* Interpret Logistic Regression coefficients.
* Investigate Decision Tree complexity and overfitting.
* Identify the best model to continue with on Day 3.

---

## Dataset

The project uses the **Adult Income dataset**.

The target variable is binary:

* `<=50K` - negative class
* `>50K` - positive class

The features include demographic, education, employment, relationship, and financial information such as:

* Age
* Education
* Education number
* Capital gain
* Capital loss
* Hours per week
* Marital status
* Occupation
* Relationship
* Sex
* Native country
* Workclass
* Race

---

## Preprocessing

A `ColumnTransformer` and pipeline-based preprocessing approach was used.

### Numerical Features

Numerical variables were:

1. Median imputed for missing values.
2. Standardized using `StandardScaler`.

### Categorical Features

Categorical variables were:

1. Missing values were handled using the most frequent category.
2. One-hot encoded using `OneHotEncoder`.

Using pipelines ensures that preprocessing is fitted only on the training data and then applied consistently to the test data.

---

## Models

Three types of approaches were evaluated:

### 1. Majority Class Baseline

A simple baseline that predicts the majority class for every observation.

### 2. Education Rule Baseline

A rule-based baseline using education-related information.

### 3. Logistic Regression

A linear classification model trained using the complete preprocessing pipeline.

### 4. Decision Tree

A Decision Tree classifier trained using the same general preprocessing framework.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC AUC
* PR AUC
* Confusion Matrix

Because the main business objective is to **reduce wasted outreach**, precision is particularly important. A false positive represents an individual predicted to earn >50K who actually belongs to the <=50K class.

---

## Model Results

| Model                   |   Accuracy |  Precision |     Recall |         F1 |    ROC AUC |     PR AUC |
| ----------------------- | ---------: | ---------: | ---------: | ---------: | ---------: | ---------: |
| Majority Class Baseline |     76.07% |      0.00% |      0.00% |      0.00% |     50.00% |        NaN |
| Education Rule Baseline |     75.30% |     48.44% |     49.70% |     49.06% |     71.89% |     44.43% |
| **Logistic Regression** | **85.28%** | **74.32%** |     58.81% | **65.66%** | **90.40%** | **76.27%** |
| Decision Tree           |     81.77% |     61.94% | **61.80%** |     61.87% |     74.92% |     47.42% |

### Best Model

**Logistic Regression** performed best overall.

It achieved:

* **85.28% Accuracy**
* **74.32% Precision**
* **58.81% Recall**
* **65.66% F1-score**
* **90.40% ROC AUC**
* **76.27% PR AUC**

Most importantly, its precision substantially exceeded the Day 1 benchmark of **48.44%**.

---

## Confusion Matrix Analysis

### Logistic Regression

|              | Predicted <=50K | Predicted >50K |
| ------------ | --------------: | -------------: |
| Actual <=50K |           6,956 |            475 |
| Actual >50K  |             963 |          1,375 |

Logistic Regression produced **475 false positives** and **963 false negatives**.

Its relatively low number of false positives helped it achieve the highest precision of **74.32%**.

### Decision Tree

|              | Predicted <=50K | Predicted >50K |
| ------------ | --------------: | -------------: |
| Actual <=50K |           6,543 |            888 |
| Actual >50K  |             893 |          1,445 |

The Decision Tree produced **888 false positives** and **893 false negatives**.

Although it detected slightly more positive cases than Logistic Regression, it generated substantially more false positives, resulting in lower precision.

---

## Logistic Regression Feature Analysis

The strongest positive coefficients included:

* `capital-gain`
* `marital-status_Married-civ-spouse`
* `marital-status_Married-AF-spouse`
* `native-country_Ireland`
* `native-country_France`
* `native-country_England`
* `occupation_Exec-managerial`
* `relationship_Wife`
* `education-num`
* `native-country_Cambodia`

The strongest negative coefficients included:

* `occupation_Priv-house-serv`
* `marital-status_Never-married`
* `native-country_Columbia`
* `sex_Female`
* `relationship_Other-relative`
* `native-country_Trinadad&Tobago`
* `occupation_Farming-fishing`
* `native-country_South`
* `occupation_Other-service`
* `relationship_Own-child`

These coefficients represent statistical associations learned by the model and should not be interpreted as causal effects.

---

## Decision Tree Analysis

The fitted Decision Tree had:

* **Depth:** 59
* **Number of leaves:** 5,100
* **Training accuracy:** 99.99%
* **Test accuracy:** 81.77%

The very high training accuracy compared with the lower test accuracy indicates substantial **overfitting**.

The first important tree splits involve:

1. `marital-status_Married-civ-spouse`
2. `capital-gain`
3. `education-num`

These features are reasonable predictors because they were also important in the earlier analysis.

---

## Model Comparison

Logistic Regression is the preferred model for the next stage of the project.

It provides:

* The highest test accuracy.
* The highest precision.
* The highest F1-score.
* The highest ROC AUC.
* The highest PR AUC.
* Fewer false positives than the Decision Tree.

The Decision Tree achieved slightly higher recall, but its much larger number of false positives and severe overfitting make it less suitable for the current business objective.

---

## Key Findings

1. The simple baseline models provide a useful reference point but have limited predictive performance.
2. Logistic Regression substantially improves upon the Day 1 precision benchmark.
3. Logistic Regression provides the strongest overall hold-out test performance.
4. Capital gain, marital status, and education-related variables are important predictors in the fitted models.
5. The Decision Tree captures nonlinear patterns but becomes excessively complex.
6. The Decision Tree's **99.99% training accuracy versus 81.77% test accuracy** is strong evidence of overfitting.
7. Reducing false positives favors Logistic Regression for the current business objective.

---

## Next Steps - Day 3

The next stage will focus primarily on improving Logistic Regression while also investigating a better-controlled Decision Tree.

Planned improvements include:

* Hyperparameter tuning.
* Class weighting.
* Alternative missing-value strategies.
* Explicit missing categories.
* Feature transformations for highly skewed variables.
* Investigation of feature redundancy.
* Decision Tree pruning and complexity control.
* Comparing the improved models using the same evaluation metrics.

---

## Conclusion

Day 2 established a complete machine learning workflow from preprocessing through model evaluation and interpretation. **Logistic Regression is currently the strongest candidate for deployment and further optimization**, particularly because it achieves substantially better precision and fewer false positives than the Decision Tree. The Decision Tree provides a useful alternative but requires complexity control because of significant overfitting.
