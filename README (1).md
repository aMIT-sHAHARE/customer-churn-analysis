<div align="center">

# Customer Churn Analysis & Prediction

**Turning e-commerce customer data into retention decisions.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Project-Portfolio%20Analysis-2ea44f)]()

</div>

## Executive summary

This project analyzes **50,000 e-commerce customer records** to understand churn behavior, segment customers using RFM analysis, and build classification models that identify customers who may leave.

The notebook compares Logistic Regression with Random Forest. On the notebook's 80/20 train-test split, the Random Forest model achieved **91.6% accuracy**, **93% precision**, **76% recall**, and **84% F1-score** for the churn class.

> **Business meaning:** The model identifies a useful group of customers for retention outreach, but it still misses approximately 24% of actual churners in the test set. It should support prioritization—not automatically decide who receives an offer.

## Business problem

Customer churn reduces repeat purchases and makes customer acquisition spend less effective. The business needs to answer:

- Which customer behaviors are associated with churn?
- Which customers should the retention team contact first?
- What interventions could address inactivity, cart abandonment, or poor support experiences?
- How well can a classification model identify churned customers?

**Objective:** Convert behavioral and transactional signals into a practical customer-retention workflow.

## Dataset overview

| Item | Details |
|---|---|
| Records | 50,000 customers |
| Features | 24 input columns plus target |
| Total columns | 25 |
| Target | `Churned` |
| Target values | `0` = not churned, `1` = churned |
| Data file | `ecommerce_customer_churn_dataset.csv` |

The dataset includes customer demographics, membership tenure, purchase behavior, website/app engagement, cart abandonment, customer-service interactions, customer value, and signup quarter.

### Important fields

| Business area | Example columns |
|---|---|
| Customer profile | `Age`, `Gender`, `Country`, `City` |
| Engagement | `Login_Frequency`, `Session_Duration_Avg`, `Pages_Per_Session`, `Mobile_App_Usage` |
| Purchase behavior | `Total_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase` |
| Retention signals | `Cart_Abandonment_Rate`, `Returns_Rate`, `Customer_Service_Calls` |
| Customer value | `Lifetime_Value`, `Credit_Balance` |
| Target | `Churned` |

## Analytical approach

### 1. Data quality and preprocessing

- Inspected dataset dimensions, data types, and null counts.
- Separated numerical and categorical columns.
- Filled missing numerical values using the median.
- Filled missing categorical values using the mode.
- Removed `Country` and `City` from the model feature set.
- Applied one-hot encoding to categorical predictors.

**Reproducibility note:** In the current notebook, imputation and one-hot encoding are performed before the train-test split. For a production-quality evaluation, fit preprocessing only on the training data using a scikit-learn `Pipeline`/`ColumnTransformer`, then transform the test data. This avoids information leakage.

### 2. Customer segmentation with RFM

RFM represents:

- **Recency:** `Days_Since_Last_Purchase`
- **Frequency:** `Total_Purchases`
- **Monetary:** `Lifetime_Value`

The notebook converts these measures into quartile scores and assigns customers to four groups:

- Best Customers
- Recent Customers
- Loyal Customers
- At Risk

The notebook's concluding summary reports an approximate segment distribution of **58% At Risk, 22% Recent, 17% Loyal, and 3% Best Customers**. Treat these as notebook-reported, rounded figures; rerun the segmentation cell to verify exact counts.

### 3. Exploratory analysis

The notebook examines churn in relation to:

- Days since last purchase
- Total purchases
- Customer-service calls
- Lifetime value
- Login frequency
- RFM customer segment

The notebook's analysis points to inactivity, lower purchase frequency, lower login activity, and customer-service interactions as useful churn signals. These are associations in this dataset, not proof that any one factor causes churn.

### 4. Model development

Two classifiers are evaluated:

- **Logistic Regression** — baseline linear classifier, configured with `class_weight="balanced"`.
- **Random Forest** — ensemble classifier, configured with 100 trees and `class_weight="balanced"`.

The notebook uses an 80/20 train-test split with `random_state=42`.

## Model evaluation

### Results recorded in the notebook

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 70.31% | **91.60%** |
| Churn precision | 48.86% | **93.00%** |
| Churn recall | 73.83% | **76.00%** |
| Churn F1-score | 58.80% | **84.00%** |

**Selected model: Random Forest**, based on the reported test-set metrics.

### Random Forest confusion matrix

| Actual / Predicted | Predicted: Not churn | Predicted: Churn |
|---|---:|---:|
| Actual: Not churn | 6,970 | 160 |
| Actual: Churn | 680 | 2,190 |

What this means on the 10,000-row test set:

- **2,190** churn cases were correctly identified.
- **680** churn cases were missed.
- **160** non-churn cases were incorrectly flagged.
- **7,130** non-churn cases were correctly identified.

A 76% churn recall means the model catches about 76 out of every 100 actual churn cases in this test set. The remaining 24 are missed, so a retention team should not rely on the model as its only churn-detection mechanism.

### Model interpretation

The notebook's Random Forest feature-importance output ranks these among the leading predictors:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `Customer_Service_Calls` | 0.1295 |
| 2 | `Lifetime_Value` | 0.1177 |
| 3 | `Cart_Abandonment_Rate` | 0.0950 |
| 4 | `Age` | 0.0789 |
| 5 | `Discount_Usage_Rate` | 0.0635 |
| 6 | `Days_Since_Last_Purchase` | 0.0566 |
| 7 | `Email_Open_Rate` | 0.0488 |
| 8 | `Total_Purchases` | 0.0487 |
| 9 | `Session_Duration_Avg` | 0.0446 |
| 10 | `Average_Order_Value` | 0.0435 |

Feature importance describes how the fitted Random Forest used features to make predictions. It does **not** establish causation, direction of effect, or the impact of changing a feature. Validate these signals with customer-level analysis and controlled experiments before making policy decisions.

## Business recommendations: from prediction to action

The model becomes useful when its output is connected to specific retention actions. The following are proposed actions based on the notebook's observed signals; they have not been tested as interventions in this project.

| Signal to monitor | Suggested business action | KPI to track |
|---|---|---|
| Long time since last purchase | Trigger a personalized win-back email or app notification; test a limited-time incentive against a no-discount message | Reactivation rate, repeat purchase rate |
| Low purchase frequency | Build replenishment reminders, product recommendations, or a loyalty journey based on prior purchases | Purchase frequency, repeat purchase rate |
| High cart abandonment | Test checkout simplification, shipping-cost transparency, and cart reminders | Cart recovery rate, checkout conversion |
| Frequent customer-service calls | Review contact reasons, resolution time, repeat contacts, and unresolved complaints; route recurring issues for follow-up | First-contact resolution, repeat contact rate, churn rate |
| Low login or app engagement | Test relevant content, saved-item reminders, and personalized recommendations | Active users, sessions per customer, conversion |
| High-value customer flagged at risk | Prioritize a human-reviewed service recovery or VIP outreach rather than sending a generic discount | Retention rate, retained revenue, offer cost |

### Recommended retention workflow

1. Score eligible customers using a validated model.
2. Create an at-risk list and rank it by churn probability and customer value.
3. Apply business rules—for example, exclude customers who cannot legally or operationally receive marketing.
4. Assign an appropriate intervention based on the likely customer problem.
5. Keep a randomized holdout group to measure incremental impact.
6. Compare churn, repeat purchase, retained revenue, and campaign cost after the observation window.
7. Refresh the model only after checking data quality and performance drift.

**Success should be measured by incremental retained customers and profit—not model accuracy alone.**

## Limitations and next steps

This is a portfolio analysis, not a production churn system. Before operational use:

- Move imputation and encoding into a train-only preprocessing pipeline to prevent leakage.
- Add feature scaling for Logistic Regression and address its convergence warning in the notebook.
- Evaluate ROC-AUC and especially PR-AUC, along with class-specific precision, recall, and F1.
- Tune the classification threshold according to retention-team capacity and the cost of false positives versus missed churners.
- Use cross-validation and compare results across different splits.
- Check class balance, calibration, and performance across relevant customer groups.
- Confirm that every feature would be available at the time a real prediction is made; remove any post-outcome or otherwise unavailable information.
- Validate RFM cutoffs and segment definitions with business stakeholders.
- Run controlled retention experiments to measure whether the proposed actions actually reduce churn.

## Repository structure

```text
customer-churn-analysis/
├── CUSTOMER_CHURN_ANALYSIS.ipynb
├── ecommerce_customer_churn_dataset.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Run the project locally

### Requirements

- Python 3.10 or later
- Jupyter Notebook
- Packages listed in `requirements.txt`

```bash
git clone https://github.com/<your-username>/customer-churn-analysis.git
cd customer-churn-analysis

python -m venv .venv
```

Activate the environment:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies and start Jupyter:

```bash
pip install -r requirements.txt
jupyter notebook
```

Open `CUSTOMER_CHURN_ANALYSIS.ipynb` and run the cells in order. Keep the CSV file in the repository root. If your notebook still contains a machine-specific absolute CSV path, replace it with:

```python
df = pd.read_csv("ecommerce_customer_churn_dataset.csv")
```

## Tools and technologies

- **Python** — analysis and model development
- **Pandas, NumPy** — data manipulation
- **Matplotlib** — visual analysis
- **Scikit-learn** — preprocessing, classification, and evaluation
- **Jupyter Notebook** — experimentation and documentation

## Author

**Amit Shahare**

Add your professional links:

- LinkedIn: `https://www.linkedin.com/in/<your-profile>/`
- GitHub: `https://github.com/<your-username>`

---

<div align="center">

**If you find this project useful, consider giving the repository a star.**

</div>
