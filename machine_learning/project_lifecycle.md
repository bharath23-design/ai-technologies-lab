# ML Project Lifecycle — Complete Reference Guide

> A concise, production-oriented reference for any ML project from idea to deployment.

---

## Table of Contents

1. [Problem Definition](#1-problem-definition)
2. [Data Collection](#2-data-collection)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis)
4. [Data Preprocessing](#4-data-preprocessing)
5. [Feature Engineering](#5-feature-engineering)
6. [Baseline Models](#6-baseline-models--quick-wins)
7. [Model Building](#7-model-building)
8. [Hyperparameter Tuning](#8-hyperparameter-tuning)
9. [Model Explainability (SHAP/LIME)](#9-model-explainability)
10. [Evaluation & Validation](#10-evaluation--validation)
11. [Production Deployment](#11-production-deployment)
12. [Monitoring & Maintenance](#12-monitoring--maintenance)

---

## 1. Problem Definition

- Define the **business objective** clearly before touching data
- Classification vs Regression vs Ranking vs Clustering?
- Set **success metrics** upfront: accuracy, F1, AUC-ROC, RMSE, business KPIs
- Define OKRs — qualitative objective + 3–5 numeric key results
- Ask: _Is ML actually needed?_ Rule-based systems or simple heuristics might suffice
- Identify **data availability**, labeling feasibility, and latency requirements

---

## 2. Data Collection

```
Data quality > Data quantity (most of the time)
```

| Source Type | Examples |
|---|---|
| Internal databases | SQL/NoSQL, data warehouses, event logs |
| APIs | Third-party services, web scraping (ethical) |
| Public datasets | Kaggle, UCI, HuggingFace Datasets, OpenML |
| Labeling | Manual annotation, Labelbox, Prodigy, weak supervision (Snorkel) |

**Best practices:**
- Document data sources, collection dates, and known biases
- Version your datasets (DVC, LakeFS, Delta Lake)
- Check for class imbalance early — use stratified sampling
- Ensure train/test splits reflect production distribution
- **~70% of ML project time goes to data work** — budget accordingly

---

## 3. Exploratory Data Analysis

```python
import pandas as pd
import ydata_profiling  # (formerly pandas-profiling)

df = pd.read_csv("data.csv")
profile = ydata_profiling.ProfileReport(df)
profile.to_file("eda_report.html")
```

**Checklist:**
- [ ] Shape, dtypes, memory usage
- [ ] Missing values (pattern: MCAR / MAR / MNAR?)
- [ ] Target distribution (imbalanced?)
- [ ] Feature distributions (skew, outliers)
- [ ] Correlations (Pearson, Spearman, Cramér's V for categoricals)
- [ ] Duplicate rows
- [ ] Cardinality of categorical features

**Tools:** `ydata-profiling`, `sweetviz`, `dtale`, `matplotlib`, `seaborn`, `plotly`

---

## 4. Data Preprocessing

### Missing Values

| Strategy | When to Use |
|---|---|
| Drop rows/columns | >50% missing, not important |
| Mean/median/mode | Random missing, numerical |
| KNN Imputer | Correlated features |
| Iterative Imputer | Complex relationships |
| Flag + impute | Missingness itself is informative |

### Outliers

- **Detection:** Z-score (>3), IQR (1.5×), Isolation Forest, DBSCAN
- **Handling:** Cap/floor (winsorize), log transform, remove, or keep (domain-dependent)

### Encoding Categoricals

| Method | When |
|---|---|
| Label Encoding | Ordinal features, tree models |
| One-Hot Encoding | Nominal, low cardinality (<15) |
| Target Encoding | High cardinality (use with CV to prevent leakage) |
| Frequency Encoding | Quick baseline |
| Binary/Hashing Encoding | Very high cardinality |

### Scaling

| Method | When |
|---|---|
| StandardScaler | Gaussian-like features, for linear models/SVM/KNN |
| MinMaxScaler | Bounded features, neural networks |
| RobustScaler | Data with outliers |
| No scaling needed | Tree-based models (XGBoost, LightGBM, CatBoost) |

---

## 5. Feature Engineering

> Feature engineering often impacts model performance **more than algorithm selection**.

### Common Techniques

- **Numerical:** log/sqrt transforms, binning, polynomial features, ratios/interactions
- **Temporal:** day_of_week, hour, is_weekend, time_since_event, rolling aggregates
- **Text:** TF-IDF, word count, sentiment score, embeddings
- **Geospatial:** distance calculations, clustering coordinates, region encoding

### Feature Selection

```python
# Correlation filter
corr_matrix = df.corr().abs()
high_corr = (corr_matrix > 0.95).sum()

# Mutual information
from sklearn.feature_selection import mutual_info_classif

# Recursive Feature Elimination
from sklearn.feature_selection import RFECV

# Model-based importance
model.feature_importances_  # tree models
```

**Methods ranked by reliability:**
1. **SHAP-based selection** — most reliable
2. **Permutation importance** — model-agnostic
3. **Built-in importance** (tree models) — fast but biased toward high-cardinality
4. **Correlation/mutual info** — quick filter

### Avoiding Data Leakage

- Never use future data to predict the past
- Always fit transformers on **train set only**, then transform test/val
- Use pipelines (`sklearn.pipeline.Pipeline`) to prevent leakage
- Be careful with target encoding — always use cross-validated encoding

### Feature Store (Production)

- Ensures consistency between training and serving
- Tools: Feast, Tecton, Hopsworks, Databricks Feature Store

---

## 6. Baseline Models — Quick Wins

### LazyPredict (Instant multi-model comparison)

```python
pip install lazypredict
```

```python
from lazypredict.Supervised import LazyClassifier, LazyRegressor

# Classification
clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)  # Sorted table of ~30 models with metrics

# Regression
reg = LazyRegressor(verbose=0, ignore_warnings=True)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
```

> Gives a quick leaderboard of ~30 sklearn models. Good for initial direction, **not for production**.

### TabPFN (Tabular Foundation Model)

```python
pip install tabpfn
```

```python
from tabpfn import TabPFNClassifier

clf = TabPFNClassifier(device="cpu")
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
proba = clf.predict_proba(X_test)
```

- **TabPFN 2.5**: State-of-the-art for tabular data in a single forward pass
- Scales to **50K samples**, **2K features**
- Beats tuned XGBoost/CatBoost without any hyperparameter tuning
- Best for: quick strong baseline, small-to-medium datasets
- Limitation: doesn't scale to very large datasets (>50K rows)

### AutoGluon (Best AutoML Framework)

```python
pip install autogluon
```

```python
from autogluon.tabular import TabularPredictor

# Quick baseline (~10 min)
predictor = TabularPredictor(label="target").fit(train_data, presets="best_quality")

# State-of-the-art (needs GPU, longer training)
predictor = TabularPredictor(label="target").fit(
    train_data,
    presets="extreme_quality"  # v1.5: uses TabPFNv2, TabICL, Mitra, RealMLP, etc.
)

predictions = predictor.predict(test_data)
leaderboard = predictor.leaderboard(test_data)
```

**AutoGluon presets:**

| Preset | Speed | Quality | Use Case |
|---|---|---|---|
| `medium_quality` | Fast | Good | Quick iteration |
| `best_quality` | Moderate | Very good | Most projects |
| `extreme_quality` | Slow (GPU) | SOTA | Competitions, critical apps |

- Fits **22+ models** including LightGBM, CatBoost, XGBoost, neural nets, and foundation models
- Auto-ensembles with multi-layer stacking
- AutoGluon 1.5 = current SOTA for tabular ML

### Other AutoML Tools

| Tool | Strengths |
|---|---|
| **AutoGluon** | Best overall, auto-ensembling, foundation models |
| **H2O AutoML** | Scalable, good for enterprise, Java-based backend |
| **FLAML** | Fast, low resource, good for constrained environments |
| **Auto-sklearn** | Bayesian optimization on sklearn |
| **PyCaret** | Low-code, good for EDA + quick modeling |
| **MLJAR** | Explanations built in, good reports |
| **Google Vertex AI AutoML** | Managed, good for GCP users |

### Recommended Baseline Strategy

```
1. LazyPredict      → Quick survey of model families (2 min)
2. TabPFN           → Strong zero-shot baseline (5 min)
3. AutoGluon        → Best ensemble baseline (10–60 min)
4. Manual tuning    → Only if AutoGluon isn't enough
```

---

## 7. Model Building

### Traditional ML Models Cheat Sheet

| Model | Type | Pros | Cons | Best For |
|---|---|---|---|---|
| **Logistic Regression** | Linear | Interpretable, fast, probabilistic | Linear decision boundary | Binary classification baseline |
| **Ridge/Lasso** | Linear | Regularization, feature selection (Lasso) | Assumes linearity | High-dimensional linear data |
| **Decision Tree** | Tree | Interpretable, no scaling needed | Overfits easily | Explainability-first tasks |
| **Random Forest** | Ensemble | Robust, handles nonlinearity | Slow inference, large model | General tabular data |
| **XGBoost** | Boosting | High accuracy, handles missing values | Many hyperparameters | Competitions, structured data |
| **LightGBM** | Boosting | Fastest tree model, memory efficient | Sensitive to overfitting on small data | Large datasets |
| **CatBoost** | Boosting | Native categorical support, less tuning | Slower training | Data with many categoricals |
| **SVM** | Kernel | Works in high dimensions | Doesn't scale (>50K rows) | Small datasets, text classification |
| **KNN** | Instance | Simple, no training phase | Slow inference, curse of dimensionality | Small data, anomaly detection |
| **Naive Bayes** | Probabilistic | Very fast, good with text | Independence assumption | Text classification, spam filtering |
| **EBM (GA²M)** | Additive | Glass-box interpretable, competitive accuracy | Slower to train | Regulated industries |

### Gradient Boosting — The Default Choice

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
```

### When to Use What

```
Small data (<1K rows)     → TabPFN, Logistic Regression, SVM
Medium data (1K–100K)     → LightGBM, XGBoost, CatBoost, AutoGluon
Large data (100K+)        → LightGBM, neural nets, distributed training
Interpretability required → EBM, Logistic Regression, Decision Tree + SHAP
Text/images/multimodal    → Transformers, AutoGluon Multimodal
```

---

## 8. Hyperparameter Tuning

### Methods (ordered by recommendation)

| Method | Description | When |
|---|---|---|
| **Optuna** | Bayesian (TPE), pruning, dashboards | Best general choice |
| **Random Search** | Random combos | Quick exploration |
| **Bayesian (Hyperopt)** | Tree Parzen Estimator | Complex search spaces |
| **Grid Search** | Exhaustive | Small search spaces only |
| **BOHB** | Bayesian + early stopping | Expensive models |

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }
    model = lgb.LGBMClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### Cross-Validation Strategies

| Strategy | When |
|---|---|
| `KFold(5)` | Default for most tabular data |
| `StratifiedKFold(5)` | Imbalanced classification |
| `TimeSeriesSplit` | Temporal data (never shuffle!) |
| `GroupKFold` | Grouped data (e.g., same user in train/test) |
| `RepeatedStratifiedKFold` | Small datasets, more stable estimates |

---

## 9. Model Explainability

### SHAP (SHapley Additive exPlanations)

```python
pip install shap
```

```python
import shap

# For tree models (fast)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test)

# Single prediction explanation
shap.waterfall_plot(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_test.iloc[0]
))

# Feature interaction
shap.dependence_plot("feature_name", shap_values, X_test)
```

**SHAP Explainer Types:**

| Explainer | For | Speed |
|---|---|---|
| `TreeExplainer` | XGBoost, LightGBM, CatBoost, RF | Fast (exact) |
| `LinearExplainer` | Linear/logistic regression | Fast |
| `DeepExplainer` | Neural networks (PyTorch/TF) | Moderate |
| `KernelExplainer` | Any model (model-agnostic) | Slow |

**Key SHAP plots:**
- **Summary plot** — global feature importance + direction of effect
- **Waterfall plot** — explains a single prediction
- **Dependence plot** — feature effect with interactions
- **Force plot** — interactive single/multi prediction view
- **Bar plot** — simple importance ranking

### LIME (Local Interpretable Model-agnostic Explanations)

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(X_train.values, feature_names=X_train.columns)
explanation = explainer.explain_instance(X_test.iloc[0].values, model.predict_proba)
explanation.show_in_notebook()
```

### SHAP vs LIME

| | SHAP | LIME |
|---|---|---|
| Theory | Solid (game theory) | Approximation |
| Consistency | Guaranteed | Not guaranteed |
| Speed | Fast for trees, slow for others | Moderate |
| Global explanations | Yes (aggregating local) | No (local only) |
| Production use | Preferred | Supplement |

### Explainability in Production

- Automate SHAP analysis in your ML pipeline
- Track **explanation drift** — if feature importances shift, investigate
- Use for model audits, fairness checks, regulatory compliance
- Log SHAP values per prediction for debugging
- Caveat: SHAP assumes feature independence — correlated features can mislead

---

## 10. Evaluation & Validation

### Classification Metrics

| Metric | When to Use |
|---|---|
| **Accuracy** | Balanced classes only |
| **Precision** | Cost of false positives is high (spam, fraud alerts) |
| **Recall** | Cost of false negatives is high (disease detection) |
| **F1 Score** | Balance precision & recall |
| **AUC-ROC** | Overall ranking ability, threshold-independent |
| **AUC-PR** | Imbalanced data (prefer over ROC) |
| **Log Loss** | When probability calibration matters |
| **MCC** | Imbalanced data, single balanced metric |

### Regression Metrics

| Metric | Notes |
|---|---|
| **MAE** | Robust to outliers, interpretable |
| **RMSE** | Penalizes large errors |
| **MAPE** | Percentage error, bad when actuals near 0 |
| **R²** | Proportion of variance explained |
| **Adjusted R²** | Accounts for number of features |

### Validation Checklist

- [ ] Cross-validation scores (not just single train/test split)
- [ ] Check for **overfitting**: train score >> val score?
- [ ] Learning curves: does more data help?
- [ ] Confusion matrix analysis
- [ ] Error analysis: **where** does the model fail?
- [ ] Calibration curve (for probability outputs)
- [ ] Test on **held-out** data that model never saw
- [ ] Check performance across subgroups (fairness)

---

## 11. Production Deployment

### Serving Patterns

| Pattern | Latency | Use Case |
|---|---|---|
| **REST API** (Flask/FastAPI) | ms | Real-time predictions |
| **Batch inference** | hours | Nightly scoring, recommendations |
| **Streaming** (Kafka + model) | sub-second | Event-driven predictions |
| **Edge deployment** (ONNX/TFLite) | μs–ms | Mobile, IoT |
| **Embedded in DB** (SQL ML) | ms | In-database scoring |

### FastAPI Example

```python
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model.joblib")

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df).max()
    return {"prediction": int(prediction[0]), "confidence": float(probability)}
```

### Model Serialization

| Format | Pros | Best For |
|---|---|---|
| `joblib` / `pickle` | Simple, sklearn native | Sklearn models |
| `ONNX` | Cross-platform, optimized inference | Production, multi-framework |
| `PMML` | Enterprise standard | Legacy systems |
| `MLflow Model` | Full lifecycle tracking | MLflow-based pipelines |

### MLOps Maturity Levels

| Level | Description |
|---|---|
| **Level 0** | Manual training, manual deployment, no monitoring |
| **Level 1** | Automated training pipeline, manual deployment, basic monitoring |
| **Level 2** | Full CI/CD for ML, automated retraining, A/B testing, full monitoring |

### Production Checklist

- [ ] Model versioning (MLflow, DVC, Weights & Biases)
- [ ] Input validation and schema enforcement
- [ ] Logging predictions + inputs for debugging
- [ ] Fallback strategy (if model fails, what happens?)
- [ ] Load testing (can it handle peak traffic?)
- [ ] A/B testing framework
- [ ] Rollback plan
- [ ] Data pipeline tests (Great Expectations, dbt tests)

### Key Tools

| Category | Tools |
|---|---|
| Experiment tracking | MLflow, W&B, Neptune, Comet |
| Pipeline orchestration | Airflow, Prefect, Dagster, Kubeflow |
| Model serving | BentoML, Seldon, TorchServe, TF Serving, Ray Serve |
| Feature store | Feast, Tecton, Hopsworks |
| Data validation | Great Expectations, Pandera, dbt |
| Containerization | Docker, Kubernetes |

---

## 12. Monitoring & Maintenance

### What to Monitor

| What | How | Tool |
|---|---|---|
| **Model performance** | Track metrics over time | Evidently, NannyML, Arize |
| **Data drift** | Compare train vs live distributions | Evidently, WhyLabs |
| **Concept drift** | Performance degrades on new patterns | NannyML (CBPE) |
| **Feature drift** | Individual feature distributions shift | Evidently, custom dashboards |
| **Prediction drift** | Output distribution changes | Histogram comparison |
| **Latency** | Response time monitoring | Prometheus + Grafana |
| **Data quality** | Schema violations, null spikes | Great Expectations |

### Retraining Triggers

```
1. Scheduled     → Retrain weekly/monthly on new data
2. Performance   → Retrain when metrics drop below threshold
3. Data drift    → Retrain when input distribution shifts significantly
4. Manual        → Domain expert identifies issues
```

### Monitoring Code Example (Evidently)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(reference_data=train_df, current_data=live_df)
report.save_html("drift_report.html")
```

---

## Quick Reference — Project Kickstart Template

```
1. Define problem & success metric
2. Collect & version data
3. EDA (ydata-profiling)
4. Preprocess (missing values, encoding, scaling)
5. Feature engineering
6. Baselines:
   a. LazyPredict  → model family survey
   b. TabPFN       → zero-shot strong baseline
   c. AutoGluon    → SOTA ensemble
7. Manual model building (if needed)
8. Hyperparameter tuning (Optuna)
9. Explain with SHAP
10. Validate thoroughly (CV + held-out + subgroups)
11. Deploy (FastAPI / batch / BentoML)
12. Monitor (Evidently / NannyML)
13. Retrain on schedule or drift trigger
```

---

## Sources & Further Reading

- [ML Lifecycle Best Practices (AIM)](https://research.aimultiple.com/machine-learning-lifecycle/)
- [7 Stages of ML Model Development (Lumenalta)](https://lumenalta.com/insights/7-stages-of-ml-model-development)
- [ML Lifecycle — Neptune.ai](https://neptune.ai/blog/life-cycle-of-a-machine-learning-project)
- [TabPFN 2.5 Model Report](https://priorlabs.ai/technical-reports/tabpfn-2-5-model-report)
- [AutoGluon Documentation](https://auto.gluon.ai/dev/tutorials/tabular/tabular-foundational-models.html)
- [AutoGluon Releases](https://github.com/autogluon/autogluon/releases)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Interpretable ML Book — SHAP Chapter](https://christophm.github.io/interpretable-ml-book/shap.html)
- [SHAP & LIME Best Practices](https://www.developerindian.com/articles/best-practices-for-model-explainability-shap-lime-in-machine-learning)
- [MLOps Guide 2026 (Glasier)](https://www.glasierinc.com/blog/machine-learning-operations-mlops-guide)
- [Google Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [Feature Engineering Guide (FeatureForm)](https://www.featureform.com/post/feature-engineering-guide)
- [ML Production Monitoring (Enlume)](https://www.enlume.com/blogs/monitoring-machine-learning-models-in-production/)
