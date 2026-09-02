# Week 1 - Day 3: Feature Engineering and Cross-Validated Model Comparison

## Engineered Features and Predictive Signal

The objective of Day 3 was to improve the Adult Census Income prediction workflow through principled feature engineering and more reliable model comparison using cross-validation. Six new features were created from existing information: age buckets, hours-per-week buckets, a capital-gain indicator, log-transformed capital gain, a higher-education indicator, and an interaction between education level and working hours.

Mutual information analysis was used to measure the univariate predictive signal of the engineered features. The strongest signals were obtained from the education-hours interaction (0.078911) and log-transformed capital gain (0.078653). Age buckets (0.058075) and the higher-education indicator (0.049926) also provided meaningful information. These results suggest that nonlinear relationships involving career stage, education, working hours, and capital gain can contribute to predicting whether an individual earns more than $50K per year.

The engineered features were integrated into a reproducible scikit-learn pipeline. Numeric features were processed using median imputation and standard scaling, while categorical features were processed using most-frequent imputation and one-hot encoding. Feature engineering was performed inside the pipeline and used only information from the current row, helping prevent information leakage during cross-validation.

## Cross-Validated Model Comparison

A 5-fold Stratified Cross-Validation strategy was used to compare Logistic Regression, Random Forest, and HistGradientBoosting. Stratification ensured that each fold maintained a similar class distribution. The hold-out test set was not used during these experiments.

HistGradientBoosting achieved the strongest overall results. It produced a mean precision of 0.7781 ± 0.0118, compared with 0.7455 ± 0.0161 for Logistic Regression and 0.7230 ± 0.0110 for Random Forest. HistGradientBoosting also achieved the highest ROC AUC of 0.9272 ± 0.0023 and the highest F1 score of 0.7080 ± 0.0089.

The cross-validation boxplots showed that HistGradientBoosting not only achieved the highest average precision but also maintained stable performance across folds. Its performance advantage suggests that nonlinear relationships in the Adult dataset are better captured by gradient boosting than by the simpler linear model or the Random Forest configuration used in this experiment.

## Statistical Comparison and Feature Importance

The top two models, HistGradientBoosting and Logistic Regression, were compared using a paired t-test on their precision scores across the five folds. The test produced a t-statistic of 14.1988 and a p-value of 0.000143. Since the p-value is substantially below 0.05, the difference in precision is statistically significant. HistGradientBoosting improved mean precision by approximately 3.26 percentage points over Logistic Regression, making the improvement practically relevant to the objective of reducing wasted outreach.

Random Forest feature importance analysis showed that the education-hours interaction was the most important engineered feature, with an importance of 0.072298. Log-transformed capital gain (0.043220) and higher education (0.023848) also contributed meaningful predictive information. These findings support the value of creating interaction and nonlinear transformations rather than relying only on the original variables.

Logistic Regression coefficient analysis also showed a strong positive association for log-transformed capital gain. Some engineered categorical features, including Mid-Career and Senior age groups, were positively associated with the positive income class, while Young, Older, and Part-Time groups showed negative associations. Differences between tree-based feature importance and Logistic Regression coefficients highlight the presence of nonlinear relationships and potential correlations between related engineered features.

## Feature Selection and Recommendation for Day 4

Feature selection was tested using SelectKBest with mutual information and k=50. The reduced feature set did not improve performance. Without feature selection, Logistic Regression achieved a precision of 0.7455, ROC AUC of 0.9125, and F1 score of 0.6731. With SelectKBest, these values decreased slightly to 0.7434, 0.9103, and 0.6636 respectively. Training time also increased from approximately 18.26 seconds to 29.20 seconds.

Therefore, the full engineered feature set will be retained. HistGradientBoostingClassifier will be the primary model selected for Day 4 hyperparameter tuning because it achieved the best and most statistically reliable performance. Logistic Regression will remain as an interpretable benchmark. The hold-out test set will continue to remain untouched until the final evaluation stage.
