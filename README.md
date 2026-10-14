# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to leave, with an evaluation process designed to avoid common pitfalls: target leakage, tuning on the test set, and over-reading small differences between models.

**Highlights**

- Compared 7 tuned classifiers with repeated stratified cross-validation, ranked on threshold-free metrics (PR-AUC, ROC-AUC).
- Removed leakage-prone columns and tested a suspect feature (`CLTV`) with and without it before excluding it.
- Handled the ~26.5% churn imbalance with stratified splits, class weights and threshold tuning, without resampling.
- Chose the decision threshold on out-of-fold training predictions only, and used the test set once for final evaluation.
- Quantified uncertainty with bootstrap confidence intervals and a paired bootstrap between the top two models.
- Final Random Forest: test **F1 0.625** (95% CI 0.587 to 0.661), precision 0.556, recall 0.714 at a threshold of 0.56.
- Interpreted the models with logistic regression coefficients, Random Forest importance, permutation importance and SHAP.

## Problem

Retaining an existing customer is usually cheaper than acquiring a new one. The goal is to estimate each customer's churn risk so a retention team can prioritise outreach.

## Data

IBM Telco Customer Churn dataset (https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7,043 customers, about 26.5% of whom churned. An 80/20 stratified split gives 5,634 training customers and a 1,409-customer test set (374 churners), both preserving the original churn rate.

Columns excluded to prevent leakage: `Churn Reason`, `Churn Score` and `Churn Value`. `CLTV` was investigated (notebook sections 1.4 and 7.5) and excluded from all final models: removing it changed repeated-CV F1 for Random Forest only from 0.651 to 0.646, and its availability at prediction time could not be verified.

## Approach

1. **Preprocessing:** a single scikit-learn `ColumnTransformer` (imputation, scaling, one-hot encoding) inside every pipeline, so nothing is fitted outside cross-validation.
2. **Class imbalance:** stratified train/test split and stratified CV folds to preserve the churn rate; `class_weight` treated as a tuned hyperparameter; evaluation on PR-AUC, ROC-AUC, precision and recall rather than accuracy. No SMOTE or undersampling, because the imbalance is moderate (about 1,500 training churners) and resampling mainly shifts the decision threshold while distorting probabilities. The test set is left at its natural distribution.
3. **Models:** Logistic Regression, Decision Tree, KNN, SVM, Random Forest, AdaBoost, XGBoost, each tuned with 5-fold grid search (churn-class F1 as the tuning metric).
4. **Model comparison:** 5-fold stratified CV repeated 3 times on the training set, with identical CLTV-free preprocessing for every model.
5. **Threshold selection:** F1-optimal threshold chosen from out-of-fold predictions on the training data, with a plateau check to see how sensitive the choice is.
6. **Final evaluation:** one pass on the held-out test set, with bootstrap confidence intervals.
7. **Interpretation and segmentation:** coefficients, feature importance, permutation importance, SHAP, and risk bands from predicted probabilities.

## Results

### Model comparison

Models were selected on cross-validation (training data). Test results are shown for reference.

| Model | CV PR-AUC (mean ± sd) | CV ROC-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|
| XGBoost | 0.691 ± 0.017 | 0.864 | 0.671 | 0.855 |
| AdaBoost | 0.683 ± 0.016 | 0.861 | 0.672 | 0.853 |
| Logistic Regression | 0.679 ± 0.018 | 0.859 | 0.642 | 0.848 |
| Random Forest | 0.678 ± 0.018 | 0.861 | 0.661 | 0.849 |
| SVM | 0.673 ± 0.016 | 0.852 | 0.629 | 0.838 |
| KNN | 0.606 ± 0.021 | 0.835 | 0.604 | 0.827 |
| Decision Tree | 0.600 ± 0.019 | 0.827 | 0.616 | 0.827 |

The top four models are statistically hard to tell apart: their PR-AUC differences are smaller than the fold-to-fold variation. KNN and the Decision Tree are clearly weaker. For context, a random ranking would score a PR-AUC of about 0.265 (the churn rate), and always predicting "no churn" gives about 0.735 accuracy, so accuracy is not informative here.

### Final model and threshold

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Random Forest | 0.56 | 0.556 | 0.714 | 0.625 | 0.849 | 0.661 |
| XGBoost | 0.36 | 0.574 | 0.719 | 0.638 | 0.855 | 0.671 |

Thresholds were chosen on out-of-fold training predictions. For Random Forest, F1 stayed within 0.01 of its peak for thresholds from 0.42 to 0.62, so the exact value is not critical. The two thresholds differ because the models output probabilities on different scales.

Paired bootstrap differences (Random Forest minus XGBoost) on the test set:

| Metric | Difference | 95% CI |
|---|---|---|
| PR-AUC | -0.011 | -0.025 to +0.003 |
| F1 (each at its own threshold) | -0.013 | -0.032 to +0.006 |
| ROC-AUC | -0.006 | -0.0115 to -0.0003 |

The PR-AUC and F1 differences are consistent with zero. The ROC-AUC gap is detectable but too small to matter in practice. The two models are treated as equivalent in predictive performance. Random Forest is used for risk segmentation (TODO: add your reason). SHAP values are computed on XGBoost, which supports the tree explainer natively and is statistically tied with Random Forest; Random Forest importance and permutation importance serve as cross-checks.

### Risk segmentation

Using Random Forest probabilities on the test set, customers are split into bands (below 30%, 30 to 60%, 60% and above): **47.0% low risk (662), 22.0% medium (310) and 31.0% high (437)**. These bands depend on the model's probability scale, so calibration should be checked before operational use.

### Key findings

Mean predicted churn probability by segment (Random Forest, test set). These are associations in observational data, not causal effects.

| Factor | Higher-risk segment | Lower-risk segment |
|---|---|---|
| Contract | Month-to-month: 42.8% | One-year: 10.2%, two-year: 3.5% |
| Tenure | 0 to 12 months: 46.5% | 61 to 72 months: 6.6% |
| Internet service | Fiber optic: 42.1% | DSL: 19.2%, none: 7.4% |
| Payment method | Electronic check: 44.7% | Mailed check: 19.3%, credit card: 16.4%, bank transfer: 16.0% |

Early-tenure and month-to-month customers are natural targets for onboarding and retention offers. Fiber-optic and electronic-check customers warrant investigation into pricing, service quality and billing experience.

## Limitations

- Thresholds are optimised for F1, which is not necessarily the business-optimal choice. There is no cost model for retention offers.
- Hyperparameters were tuned with `CLTV` included and were not re-tuned after it was removed.
- Probabilities from the class-weighted Random Forest are not calibrated, so risk bands depend on its probability scale.
- A small number of missing `Total Charges` values are median-imputed inside the pipeline.
- A single train/test split with 374 test churners limits precision; bootstrap intervals capture test-set sampling variation but not retraining variation.
- One dataset from one telecom company, so results may not transfer to other markets or time periods.
- Feature importance and SHAP values describe associations in the data, not causal effects.

## Planned improvements

- Cost-based threshold using assumed offer cost, customer value and offer success rate.
- Lift/gain table and calibration plot.
- Uplift modelling to separate "who will churn" from "who will respond to an offer".
- Small Streamlit app to score a customer and show a SHAP explanation.

## How to run

```bash
git clone <https://github.com/markreji2027/telecom-churn-intelligence>
cd <https://github.com/markreji2027/telecom-churn-intelligence>
pip install pandas numpy scikit-learn xgboost shap matplotlib jupyter
jupyter notebook notebooks
```

Use Kernel → Restart & Run All to reproduce the results. Random seeds are fixed in the notebook.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, Jupyter.

## Author

Mark Reji

LinkedIn - https://www.linkedin.com/in/mark-reji-29488b246

Email - markreji2027@gmail.com
