# ⚽ FIFA Unified Scouting System — ML Pipeline

> A complete machine learning pipeline built on FIFA player data, progressing from baseline regression & classification models to an intelligent, ensemble-powered unified scouting system.

---

## 📌 Project Overview

This project is structured across **two phases** (Assignment 2 and Assignment 3), both working on the same FIFA dataset. Together, they form a production-ready scouting system that:

- **Predicts player market value** (`Value Per M$`) — Regression task
- **Classifies player performance level** (`Overall_Rating` → 4 tiers) — Classification task

The project evolves from individual baseline models to a fully unified pipeline with ensemble learning, hyperparameter tuning, and cross-validation.

---

## 📁 Dataset

**File:** `Fifa.csv`

| Feature | Type | Description |
|---|---|---|
| `Name` | Categorical | Player name (dropped after use) |
| `Country` | Categorical | Player nationality (converted to continent) |
| `Team` | Categorical | Club or national team (engineered to binary flags) |
| `Position` | Categorical | Field position (GK, FW, MF, DF…) |
| `Age` | Numerical | Player age |
| `Overall_Rating` | Numerical | FIFA overall skill rating (also used as classification source) |
| `Future Potential` | Numerical | Future rating potential |
| `Total_Stats Score` | Numerical | Aggregate of all FIFA skill attributes |
| `Value Per M$` | Numerical | Market value in millions USD (main regression target) |

---

## 🗂️ Project Structure

```
FIFA_Scouting_System/
│
├── Assignment_2_.ipynb       # Phase 1: Baseline Models
├── Assignment_3_.ipynb       # Phase 2: Unified Scouting System
├── Fifa.csv                  # Raw dataset
└── README.md
```

---

## 🔬 Phase 1 — Assignment 2: Baseline Models

### Task 1 — EDA
- Explored shape, types, missing values, and duplicates
- Visualized market value distribution (skewness ≈ 7.98 → highly right-skewed)
- Built a correlation heatmap to identify predictors
- Analyzed average player rating per position

### Task 2 — Data Preprocessing
- **Feature engineering:** country names standardized and mapped to continents (Africa, Asia, Europe, North America, South America, Oceania)
- **Team classification:** `Team` column converted to binary flags — `is_national_team` and `is_club` — using country name matching and club keyword detection
- **Encoding:** One-Hot Encoding for `continent` and `Position`; `Name` and `Country` dropped
- **Outlier handling:** IQR clipping on all numerical features
- **Scaling:** StandardScaler applied to age, potential, and stats features

### Task 3 — Target Variables
Two targets created from the data:

- **Regression target (`y_reg`):** `log1p(Value Per M$)` — log transformation applied to reduce skewness
- **Classification target (`y_cls`):** `Overall_Rating` binned into 4 quartile-based performance classes:
  - `0` → Low (≤ Q1)
  - `1` → Mid (Q1–Q2)
  - `2` → High (Q2–Q3)
  - `3` → Elite (> Q3)

### Task 4 — Polynomial Regression
- Tested degrees 1–4; degree 4 achieved best R² (~0.958 train / ~0.958 test)
- **Ridge Regression** selected as best model — Best Alpha = 0.1, Test RMSE = 0.1552
- **Lasso Regression** — Best Alpha = 0.001, Test RMSE = 0.1600; removed 19 polynomial features
- Ridge outperformed Lasso: retaining all features with L2 shrinkage was more effective than L1 feature elimination

### Task 5 — Logistic Regression
- Trained without `Overall_Rating` to prevent data leakage
- Hyperparameter tuning via validation curve on `C` (regularization strength)
- **L2 Accuracy: 84.3%** vs L1 Accuracy: 83.9%
- Most confusion between neighboring classes (Low↔Mid, High↔Elite)

### Task 6 — Naïve Bayes
Three variants compared:

| Model | Features Used | Notes |
|---|---|---|
| GaussianNB | Numerical (Age, Potential, Stats) | Best fit for continuous features |
| BernoulliNB | One-hot encoded features | Works with binary inputs |
| ComplementNB | Non-negative OHE features | Designed for imbalanced data |

GaussianNB was identified as the most appropriate variant. Scaling had minimal effect on GaussianNB since it relies on distribution parameters, not distances.

### Task 7 — Cross-Validation

**Regression (Ridge, 5-Fold KFold):**
- Mean RMSE: 0.1595 | Std: 0.0180

**Classification (Stratified KFold):**
- Logistic Regression consistently outperformed Naïve Bayes in accuracy and stability

### Task 8 — Analysis & Discussion
- Ridge outperforms Lasso on OHE-heavy datasets (L1 eliminates correlated binary features)
- Classification is easier than regression on this dataset — predicting tiers vs. exact continuous values
- Increasing alpha in Ridge initially helps, then causes underfitting; Lasso continuously degrades

---

## 🚀 Phase 2 — Assignment 3: Unified Scouting System

This phase evolves the system from individual models to a **structured, robust, unified pipeline**.

### Unified Preprocessing Pipeline

Built using scikit-learn's `Pipeline` + `ColumnTransformer`:

- **Numerical pipeline:** `SimpleImputer(median)` → `StandardScaler`
- **Categorical pipeline:** `SimpleImputer(mode)` → `OneHotEncoder(handle_unknown='ignore')`
- Duplicate removal applied to both train and test sets
- IQR clipping on: Age, Future Potential, Total_Stats Score, Overall_Rating

### Advanced Models (Baseline)

| Model | Task | Train Acc/R² | Test Acc/R² |
|---|---|---|---|
| KNN Regressor | Regression | 0.9363 | 0.9363 |
| KNN Classifier | Classification | ~high | 82.57% |
| Random Forest | Classification | 99.79% | 84.09% |
| SVM (RBF kernel) | Classification | 86.77% | 82.36% |

### Model Selection Rationale

Three architecturally distinct learners were chosen to cover diverse learning approaches:

- **KNN** — instance-based; captures local similarity between players
- **SVM** — kernel-based; handles high-dimensional OHE features and non-linear boundaries
- **Random Forest** — tree-based ensemble; handles mixed feature types and complex interactions

### Hyperparameter Optimization (GridSearchCV)

| Model | Best Parameters | Test Score After | CV Score |
|---|---|---|---|
| Random Forest | n_estimators=300, max_depth=20, max_features='sqrt' | 84.14% | 84.86% |
| SVM | C=1, gamma='scale', kernel='rbf' | 82.36% | 85.61% |
| KNN Classifier | n_neighbors=7, weights='uniform', metric='euclidean' | 83.01% | 79.69% |
| KNN Regressor | n_neighbors=11, weights='uniform', metric='euclidean' | R²=0.9426 | R²=0.9127 |

### Ensemble Learning

Two ensemble strategies applied to both tasks:

**Classification:**

| Method | Test Accuracy | CV Score |
|---|---|---|
| Hard Voting (RF + SVM + KNN) | 84.35% | 84.07% |
| Stacking (meta: Logistic Regression) | 83.04% | 85.57% |

**Regression:**

| Method | Test R² | CV R² |
|---|---|---|
| Voting Regressor (KNN) | 0.9426 | 0.9103 |
| Stacking Regressor (meta: Linear Regression) | 0.9410 | 0.9105 |

### Cross-Validation Summary

- Classification (Stratified KFold): All models stable, low std across folds
- Regression (KFold): Mean R² ≈ 0.91, stable across folds

### Unified Scouting System

The final system takes a player's profile as input and returns two outputs in one call:

```python
result = unified_scouting_system(player_data)
# → {"performance_tier": 1, "estimated_value": 0.1271}
```

Tier labels: `0 = Low`, `1 = Mid`, `2 = High`, `3 = Elite`

**Stability:** Mean CV Accuracy = 85.61%, Std = 0.0046 — indicating highly stable predictions.

**Bias-Variance:** Training Accuracy 86.77% vs Test Accuracy 82.36% — small gap, balanced tradeoff.

---

## 📊 Results Summary

### Regression (Best Model: Ridge, Phase 1 | KNN Optimized, Phase 2)

| Phase | Model | Test RMSE | Test R² |
|---|---|---|---|
| Phase 1 | Ridge (degree 4) | 0.1552 | — |
| Phase 2 | KNN Regressor (optimized) | 0.1106 | 0.9426 |

### Classification (Best Model: Logistic Regression, Phase 1 | Voting Classifier, Phase 2)

| Phase | Model | Test Accuracy |
|---|---|---|
| Phase 1 | Logistic Regression (L2) | 84.3% |
| Phase 2 | Voting Classifier (RF+SVM+KNN) | 84.35% |
| Phase 2 | Stacking (CV) | 85.57% |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | Full ML pipeline (preprocessing, models, evaluation) |
| `pycountry`, `pycountry_convert` | Country → continent mapping |

---

## 👩‍💼 Author

**Shahd Ahmed Farghaly**
*Data Science Student — Alexandria University*

📧 [shahdfarghaly2005@gmail.com](mailto:shahdfarghaly2005@gmail.com)
🔗 [LinkedIn Profile](https://www.linkedin.com/in/shahd-farghaly-bb9356332)
