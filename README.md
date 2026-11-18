# Telecom Customer Churn Prediction & Retention ROI Simulation

An end-to-end machine learning project that identifies high-risk telecom customers before they churn and translates model predictions into a quantified, budget-constrained customer retention strategy.

---

## Project Overview

Acquiring new telecom subscribers costs significantly more than retaining existing ones. Rather than applying a generic `0.50` probability threshold—which ignores marketing and call-center capacity limits—this project aligns model evaluation directly with operational constraints by ranking customers by churn risk and targeting the **Top 20% highest-risk segment** (`Precision@Top20%` and `Lift@Top20%`).

### Key Objectives
1. **Engineer Behavioral & Service Signals:** Derive domain-specific ratios capturing handset aging, billing shock, and network call quality.
2. **Prevent Data Leakage:** Implement a strict `70% / 15% / 15%` stratified train/validation/test split with all imputations and encodings encapsulated inside a scikit-learn `Pipeline`.
3. **Benchmark Tree-Based Models:** Compare **Random Forest**, **XGBoost**, and **LightGBM** against a baseline **Dummy Classifier** using business-aligned ranking metrics.
4. **Quantify Financial Impact (ROI):** Simulate a targeted retention campaign across the full customer base, including a sensitivity analysis across multiple retention success rates (`20%`, `30%`, and `40%`).

---

## Project Workflow & Methodology

### 1. Feature Engineering
Three domain-informed ratios were engineered using safe division handling (`safe_ratio`) to avoid zero-division artifacts:
* **`eqp_to_tenure_ratio`**: Ratio of current equipment age (`eqpdays`) to total customer tenure (`months × 30`), capturing device upgrade fatigue.
* **`overage_rev_ratio`**: Share of total monthly revenue coming from overage charges (`ovrrev_Mean / rev_Mean`), capturing unexpected billing friction.
* **`call_failure_rate`**: Ratio of dropped/blocked calls to total call attempts (`drop_blk_Mean / attempt_Mean`), measuring network experience.

### 2. Preprocessing & Modeling Pipeline
* **Data Splitting:** Stratified split into **70% Training**, **15% Validation**, and **15% Held-Out Test** sets.
* **ColumnTransformer Architecture:**
  * *Numerical Features:* Median imputation (`SimpleImputer(strategy="median")`).
  * *Categorical Features:* Constant `"Missing"` imputation followed by `OneHotEncoder(handle_unknown="ignore")`.
* **Candidate Models:** `DummyClassifier` (baseline), `RandomForestClassifier`, `XGBClassifier`, and `LGBMClassifier`.

### 3. Model Selection & Business Metrics
Models were ranked on validation data primarily by **`Precision@Top20%`** and **`ROC-AUC`**, then retrained on the combined training + validation sets (`85%`) for final evaluation on the unseen test set (`15%`).

| Metric | Test Set Result | Business Interpretation |
| :--- | :---: | :--- |
| **Selected Model** | `[e.g., LightGBM / XGBoost]` | Best validation `Precision@Top20%` & `ROC-AUC` |
| **Precision@Top20%** | `[XX.XX%]` | Share of targeted high-risk customers who actually churn |
| **Recall@Top20%** | `[XX.XX%]` | Share of all company churners captured within the top 20% |
| **Lift@Top20%** | `[X.XXx]` | Improvement over random customer targeting |
| **ROC-AUC** | `[0.XXXX]` | Overall discrimination power across thresholds |

---

## Business Proposal & ROI Simulation

To evaluate financial feasibility for executive stakeholders, the model's test-set performance was projected onto the full customer base (`[100,000]` customers) under transparent financial assumptions:
* **Target Segment:** Top `20%` highest-risk customers (`[20,000]` customers)
* **Retention Offer Cost:** `$50.00` per targeted customer (handset upgrade credit / bonus bundle)
* **Revenue Horizon:** `12 months` of retained average monthly revenue (`ARPU`)

### Sensitivity Analysis (1-Year Horizon)

| Retention Success Rate | Retained Customers | Gross Retained Revenue | Campaign Cost | Net Financial Benefit | Scenario ROI |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **20% (Conservative)** | `[X,XXX]` | `$[X,XXX,XXX]` | `$1,000,000` | `$[X,XXX,XXX]` | `[XXX.X%]` |
| **30% (Base Case)** | `[X,XXX]` | `$[X,XXX,XXX]` | `$1,000,000` | `$[X,XXX,XXX]` | `[XXX.X%]` |
| **40% (Optimistic)** | `[X,XXX]` | `$[X,XXX,XXX]` | `$1,000,000` | `$[X,XXX,XXX]` | `[XXX.X%]` |

---

## Repository Structure

```text
├── churn_prediction_roi.ipynb   # Main notebook: EDA, feature engineering, modeling & ROI simulation
├── README.md                    # Project documentation and business summary
└── requirements.txt             # Python dependencies