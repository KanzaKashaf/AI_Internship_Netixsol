# Week 1 - Day 1: Adult Census Income Prediction

## Problem Framing

The objective of this project is to predict whether an individual earns more than $50,000 per year using demographic and employment-related features from the UCI Adult Census Income dataset. In a targeted marketing scenario, these predictions could help a business focus its outreach on individuals who are more likely to belong to a higher-income segment. The positive class represents individuals earning more than $50K per year.

The dataset contains 48,842 observations. The positive class accounts for **23.93%** of the dataset, while the remaining observations belong to the negative class. This class imbalance means that accuracy alone can be misleading.

## Chosen Metric

The primary metric selected for this project is **Precision**. In the targeted marketing scenario, incorrectly identifying a low-income individual as high-income could result in wasted outreach and unnecessary business costs. Therefore, the goal is to ensure that individuals predicted as high-income are genuinely likely to earn more than $50K.

Although precision is the primary metric, recall, F1-score, ROC AUC, and PR AUC will also be monitored to ensure that improvements in precision do not result in missing too many genuinely high-income individuals.

## Baseline Results

Two simple baselines were evaluated on a stratified 20% hold-out test set.

| Model                   | Accuracy | Precision | Recall |     F1 | ROC AUC | PR AUC |
| ----------------------- | -------: | --------: | -----: | -----: | ------: | -----: |
| Majority-Class Baseline |   76.07% |     0.00% |  0.00% |  0.00% |  50.00% |    -   |
| Education Rule Baseline |   75.30% |    48.44% | 49.70% | 49.06% |  71.89% | 44.43% |

The Majority-Class Baseline achieved higher accuracy by predicting every individual as belonging to the most frequent negative class. However, it failed completely to identify high-income individuals, demonstrating why accuracy is not sufficient for this problem. The Education Rule Baseline, which predicts high income when **education-num ≥ 13**, performed substantially better on the metrics relevant to identifying the positive class.

## Initial Error Analysis

The Education Rule Baseline produced **1,237 false positives** and **1,176 false negatives**. Many false positives had higher educational attainment but still earned less than $50K, showing that education alone is not sufficient to determine income. False negatives had education levels below the selected threshold but still earned more than $50K; several worked in occupations such as executive managerial, sales, and craft repair and often worked longer hours.

The analysis suggests that additional features such as **age, occupation, marital status, hours-per-week, capital-gain, and capital-loss** may improve predictions. Future work should address missing categorical values, categorical encoding, skewed capital-related features, class imbalance, and potential redundancy between education and education-num.

## Next Steps

For the remainder of the week, **Precision will remain the primary optimization metric**. The current Education Rule Baseline achieved **48.44% precision**, providing a clear minimum benchmark for future machine learning models. The next models should aim to improve precision beyond this baseline while maintaining useful recall and improving overall ranking performance.
