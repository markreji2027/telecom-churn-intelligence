# Telecom Customer Churn Prediction

Predicting which telecom customers are likely to leave, with an evaluation process designed to avoid the usual pitfalls: target leakage, tuning on the test set, and over-reading small differences between models.

**Highlights**

- Compared 7 tuned classifiers with repeated stratified cross-validation, ranked on threshold-free metrics (PR-AUC, ROC-AUC).
- Removed leakage-prone columns and tested a suspect feature (`CLTV`) with and without it before excluding it.
- Chose the decision threshold on out-of-fold training predictions only, and touched the test set once.
- Quantified uncertainty with bootstrap confidence intervals and a paired bootstrap between the top two models.
- Final Random Forest: test **F1 0.625** (95% CI 0.587 to 0.661), precision 0.556, recall 0.714 at a threshold of 0.56.
- Interpreted the model with logistic regression coefficients, Random Forest importance, permutation importance and SHAP.

## Problem

Retaining an existing customer is usually cheaper than acquiring a new one. The goal is to estimate each customer's churn risk so a retention team can prioritise outreach.

## Data

IBM Telco Customer Churn dataset (TODO: add link): 7,043 customers, about 26.5% of whom churned. The held-out test set has 1,409 customers (374 churners).

Columns excluded to prevent leakage: `Churn Reason`, `Churn Score` and `Churn Value`. `CLTV` was investigated (see the notebook, sections 1.4 and 7.5) and excluded from all final models.

## Approach

1. **Preprocessing:** a single scikit-learn `ColumnTransformer` (imputation, scaling, one-hot encoding) inside every pipeline, so nothing is fitted outside cross-validation.
2. **Models:** Logistic Regression, Decision Tree, KNN, SVM, Random Forest, AdaBoost, XGBoost, each tuned with 5-fold grid search (F1 as the tuning metric).
3. **Model comparison:** 5-fold stratified CV repeated 3 times on the training set, with identical preprocessing for every model.
4. **Threshold selection:** F1-optimal threshold chosen from out-of-fold predictions on the training data, with a plateau check to see how sensitive the choice is.
5. **Final evaluation:** one pass on the held-out test set, with bootstrap confidence intervals.
6. **Interpretation and segmentation:** coefficients, feature importance, permutation importance, SHAP, and risk bands from predicted probabilities.

## Results

### Model comparison

Models were selected on cross-validation (training data). Test results are reported for reference at each model's default cutoff.

| Model | CV PR-AUC (mean ± sd) | CV ROC-AUC | Test PR-AUC | Test ROC-AUC |
|---|---|---|---|---|
| XGBoost | 0.691 ± 0.017 | 0.864 | 0.671 | 0.855 |
| AdaBoost | 0.683 ± 0.016 | 0.861 | 0.672 | 0.853 |
| Logistic Regression | 0.679 ± 0.018 | 0.859 | 0.642 | 0.848 |
| Random Forest | 0.678 ± 0.018 | 0.861 | 0.661 | 0.849 |
| SVM | 0.673 ± 0.016 | 0.852 | 0.629 | 0.838 |
| KNN | 0.606 ± 0.021 | 0.835 | 0.604 | 0.827 |
| Decision Tree | 0.600 ± 0.019 | 0.827 | 0.616 | 0.827 |

The top four models are statistically hard to tell apart: their PR-AUC differences are smaller than the fold-to-fold variation. KNN and the Decision Tree are clearly weaker. Accuracy is not informative here, since always predicting "no churn" scores about 0.735.

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

PR-AUC and F1 differences are consistent with zero. The ROC-AUC gap is detectable but too small to matter in practice. The two models are treated as equivalent in predictive performance. Random Forest is used for interpretation and segmentation (TODO: add your reason).

### Risk segmentation

Using Random Forest probabilities on the test set, customers are split into bands (<30%, 30 to 60%, 60% and above): **47.0% low risk, 22.0% medium, 31.0% high**.

### Key findings

TODO: add 2 to 3 business findings from your SHAP and segmentation sections (for example, which contract types, tenure ranges or services are most associated with churn).

## Limitations

- Thresholds are optimised for F1, which is not necessarily the business-optimal choice. There is no cost model for retention offers.
- Hyperparameters were tuned with `CLTV` included and were not re-tuned after it was removed.
- Probabilities from the class-weighted Random Forest are not calibrated, so risk bands depend on its probability scale.
- A single train/test split with 374 test churners limits precision; bootstrap intervals capture test-set sampling variation but not retraining variation.
- Feature importance and SHAP values describe associations in the data, not causal effects.

## Planned improvements

- Cost-based threshold using assumed offer cost, customer value and offer success rate.
- Lift/gain table and calibration plot.
- Uplift modelling to separate "who will churn" from "who will respond to an offer".
- Small Streamlit app to score a customer and show a SHAP explanation.

## Repository structure

TODO: adjust to match your repo.

```
.
├── data/                      # dataset (or instructions to download it)
├── notebooks/
│   └── 03_ml_models.ipynb     # modelling, evaluation, interpretation
└── README.md
```

## How to run

```bash
git clone <your-repo-url>
cd <your-repo>
pip install pandas numpy scikit-learn xgboost shap matplotlib jupyter
jupyter notebook notebooks/03_ml_models.ipynb
```

Use Kernel → Restart & Run All to reproduce the results. Random seeds are fixed in the notebook.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, matplotlib, Jupyter.

## Author

Mark Reji

LinkedIn : https://www.linkedin.com/in/mark-reji-29488b246
Email : markreji2027@gmail.com
