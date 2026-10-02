# Customer Churn Prediction & Analysis

An end-to-end machine learning project for analyzing and predicting customer churn using a telecom customer dataset containing **7,043 customer records**.

The project covers data cleaning, exploratory data analysis, preprocessing, machine learning, hyperparameter tuning, model evaluation, model interpretation, threshold analysis, and customer risk segmentation.

---

## 📌 Project Overview

Customer churn is an important business problem for subscription-based companies because retaining existing customers can be more efficient than acquiring new ones.

This project aims to:

* Identify patterns associated with customer churn
* Build machine learning models to predict churn
* Compare multiple classification algorithms
* Tune model hyperparameters using cross-validation
* Evaluate models using multiple classification metrics
* Interpret model predictions using feature importance and SHAP
* Analyze different churn probability thresholds
* Segment customers into low-, medium-, and high-risk groups
* Translate analytical findings into potential business actions

> **Note:** The identified relationships represent predictive associations and should not be interpreted as causal relationships.

---

## 📊 Dataset

The project uses the IBM Telco Customer Churn dataset.

* **Customers:** 7,043
* **Target:** Churn Label
* **Features:** Customer demographics, tenure, services, contract information, billing information, and customer lifetime value
* **Target classes:** Churned (`Yes`) and Not Churned (`No`)

The original dataset contains additional fields related to churn outcomes and geographic information. Variables that could introduce target leakage or were outside the scope of the predictive analysis were excluded during preprocessing.

---

## 🔍 Exploratory Data Analysis

The exploratory analysis examined:

* Churn distribution
* Customer tenure
* Monthly charges
* Total charges
* Customer lifetime value (CLTV)
* Contract type
* Internet service
* Payment method
* Technical support
* Online security
* Dependents
* Senior citizen status
* Other customer service characteristics

Some notable patterns observed in the dataset include higher churn rates among:

* Month-to-month contract customers
* Customers with shorter tenure
* Fiber-optic internet customers
* Electronic-check payment customers
* Customers without online security or technical support

These patterns describe associations within the dataset and do not establish causation.

---

## ⚙️ Machine Learning Workflow

The project follows this general workflow:

```text
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Train-Test Split
   ↓
Feature Preparation
   ↓
Categorical Encoding + Numerical Scaling
   ↓
Baseline Models
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Model Interpretation
   ↓
Threshold Analysis
   ↓
Customer Risk Segmentation
   ↓
Business Insights
```

### Preprocessing

Categorical variables were transformed using **One-Hot Encoding**.

Numerical variables were standardized using **StandardScaler**.

A stratified train-test split was used to preserve the original churn proportion in both datasets.

---

## 🤖 Models

Seven classification algorithms were developed and evaluated:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. AdaBoost
7. XGBoost

Hyperparameters were optimized using **5-fold GridSearchCV**.

The primary tuning metric was the **F1-score for the churn class**.

---

## 📈 Model Performance

The final models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Confusion matrices

### Test Set Performance

| Model               |   Accuracy |  Precision |     Recall |         F1 |
| ------------------- | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     74.38% |     51.15% | **77.27%** |     61.55% |
| Decision Tree       |     78.21% |     60.37% |     52.14% |     55.95% |
| KNN                 |     78.71% |     59.89% |     59.89% |     59.89% |
| SVM                 |     79.99% |     64.29% |     55.35% |     59.48% |
| Random Forest       |     76.37% |     53.88% |     76.20% | **63.12%** |
| AdaBoost            |     80.70% |     65.45% |     57.75% |     61.36% |
| XGBoost             | **81.05%** | **67.09%** |     56.15% |     61.14% |

At the default 0.50 classification threshold:

* **XGBoost** achieved the highest accuracy at **81.05%**
* **XGBoost** achieved the highest ROC-AUC at **0.856**
* **Random Forest** achieved the highest F1-score at **63.12%**
* **Logistic Regression** achieved the highest churn recall at **77.27%**
* **AdaBoost** achieved the highest PR-AUC at **0.674**

These results demonstrate that model performance depends on the evaluation metric and business objective rather than a single universal metric.

---

## 📉 ROC-AUC and PR-AUC

| Model               |    ROC-AUC |     PR-AUC |
| ------------------- | ---------: | ---------: |
| Logistic Regression |     0.8485 |     0.6442 |
| KNN                 |     0.8245 |     0.6016 |
| SVM                 |     0.8372 |     0.6279 |
| Decision Tree       |     0.8257 |     0.5894 |
| Random Forest       |     0.8498 |     0.6560 |
| AdaBoost            | **0.8527** | **0.6742** |
| XGBoost             | **0.8561** |     0.6718 |

ROC curves for all seven tuned models are included in the machine learning notebook.

---

## 🔎 Model Interpretation

Several approaches were used to understand model behavior.

### Logistic Regression Coefficients

The logistic regression coefficients were examined to understand the direction of association between encoded features and predicted churn.

Important variables included:

* Tenure
* Contract type
* Dependents
* Internet service
* Total charges
* Payment method

### Random Forest Feature Importance

Tree-based feature importance highlighted variables such as:

* Contract
* Tenure
* Total charges
* Monthly charges
* Online security
* CLTV
* Internet service

### Permutation Feature Importance

Permutation importance was calculated using the original feature columns to evaluate how much model performance changed when individual features were shuffled.

The most influential features included:

* Contract
* Dependents
* Internet service
* Tenure
* Total charges

### SHAP Analysis

SHAP was used with the XGBoost model to examine both global and individual predictions.

The analysis showed that factors such as:

* Month-to-month contracts
* Shorter tenure
* Higher monthly charges
* Lack of dependents
* Fiber-optic service
* Lack of online security
* Lack of technical support
* Electronic-check payment

could contribute toward higher predicted churn risk, while longer tenure and two-year contracts generally contributed in the opposite direction.

---

## 🎯 Threshold Analysis

The project also examined how changing the classification threshold affects the trade-off between precision and recall.

For example, lowering the threshold generally increases the number of customers identified as potential churners, increasing recall but also increasing false positives.

This demonstrates why a classification threshold should be selected according to the **business cost of false positives versus false negatives**, rather than automatically assuming that 0.50 is optimal.

---

## 🚦 Customer Risk Segmentation

The XGBoost model was used to calculate predicted churn probabilities for the test customers.

Analytical risk bands were defined as:

| Risk Category | Churn Probability |    Customers |
| ------------- | ----------------: | -----------: |
| Low           |            < 0.30 | 877 (62.24%) |
| Medium        |     0.30 – < 0.60 | 336 (23.85%) |
| High          |            ≥ 0.60 | 196 (13.91%) |

These thresholds are analytical categories created for this project and are not universal business standards.

### Risk Patterns

Average predicted churn probability was higher among:

* Month-to-month customers
* Customers with shorter tenure
* Fiber-optic customers
* Electronic-check customers

The predicted risk decreased substantially as customer tenure increased.

---

## 💡 Business Insights

The analysis suggests several areas that could be investigated further:

### 1. Early-Tenure Customers

Customers in their first year showed substantially higher predicted churn risk.

Potential focus:

* Customer onboarding
* Early engagement
* Support quality
* Service adoption
* Early retention programs

### 2. Month-to-Month Contracts

Month-to-month customers showed substantially higher predicted churn risk than customers with longer contracts.

Potential areas for further investigation include:

* Pricing
* Contract flexibility
* Retention offers
* Customer expectations
* Competitive alternatives

### 3. Fiber-Optic Customers

The fiber-optic segment showed higher predicted churn risk.

Further investigation could examine:

* Service quality
* Pricing
* Technical issues
* Customer expectations
* Competitive offerings

### 4. Electronic-Check Customers

Customers using electronic checks showed higher predicted churn risk.

The company could investigate whether payment experience, payment friction, or other characteristics of this customer segment are associated with retention.

These recommendations are based on predictive associations and should be validated using additional business data before implementation.

---

## ⚖️ Handling Class Imbalance

Approximately **26.5%** of customers in the dataset belonged to the churn class.

The project addressed this by:

* Using a stratified train-test split
* Evaluating churn-class Precision, Recall and F1
* Using churn-class F1 as the primary GridSearchCV scoring metric
* Testing `class_weight="balanced"` during hyperparameter tuning for selected models

SMOTE and undersampling were not used.

---

## 📁 Project Structure

```text
customer-churn-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   └── Customer_Churn_Project.docx
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   └── 03_ml_models.ipynb
│
├── reports/
│   └── figures/
│
├── src/
│
├── models/
│
├── app/
│
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

### Programming & Analysis

* Python
* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* SHAP

### Development

* Jupyter Notebook
* VS Code
* Git
* GitHub

---

## ▶️ How to Run

Clone the repository:

```bash
git clone https://github.com/markreji2027/customer-churn-prediction.git
cd customer-churn-prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Open the notebooks in Jupyter Notebook or VS Code and run them sequentially:

```text
01_data_understanding.ipynb
        ↓
02_exploratory_data_analysis.ipynb
        ↓
03_ml_models.ipynb
```

---

## ⚠️ Limitations

* The analysis is based on a single telecom customer dataset.
* The dataset may not represent future or external customer populations.
* Several potentially important business variables are unavailable, including complaints, service outages, customer satisfaction, actual usage, competitor activity, and pricing changes.
* Predictive associations do not establish causal relationships.
* Probability thresholds should be validated using business costs and future validation data before deployment.
* Predicted churn represents model-estimated risk rather than a guaranteed customer outcome.

---

## 📌 Key Takeaway

This project demonstrates an end-to-end approach to customer churn prediction, combining **statistical analysis, machine learning, model evaluation, explainability, and business-oriented risk segmentation**.

Rather than relying on a single model or metric, the project evaluates multiple algorithms and examines how model performance changes across different evaluation criteria and classification thresholds.
