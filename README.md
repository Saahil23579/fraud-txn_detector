# Fraud Detection — Project Setup Guide

## Folder Structure

```
fraud-detection/
│
├── data/                                  ← Place your raw dataset here
│   └── fraud_dataset_10000_records.csv
│
├── src/                                   ← Source code
│   └── fraud.py
|   └── fraud_api.py
│
├── models/                                ← Saved model artefacts (auto-generated)
│   ├── fraud_model.pkl
│   └── model_meta.json
│
├── outputs/                               ← All plots & reports (auto-generated)
│   ├── 1_model_comparison.png
│   ├── 2_confusion_matrices.png
│   ├── 3_roc_curves.png
│   ├── 4_feature_importance.png
│   ├── 5_threshold_analysis.png
│   └── 6_shap_explainability.png
│
├── requirements.txt                       ← Python dependencies
└── README.md
```

---

## Installation

### 1 — Create a virtual environment (recommended)

```bash
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows
.venv\Scripts\activate
```

### 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### 3 — Run the pipeline

Make sure your CSV is inside the `data/` folder, then:

```bash
cd src
python fraud_api.py
```

> **Note:** If you moved the CSV to `data/`, update line 55 in `fraud.py`:
> ```python
> # Before
> df = pd.read_csv("fraud_dataset_10000_records.csv")
> # After
> df = pd.read_csv("../data/fraud_dataset_10000_records.csv")
> ```

---

## Library Reference

| Library            | Version   | Purpose                                      |
|--------------------|-----------|----------------------------------------------|
| numpy              | >=1.24.0  | Numerical operations, cyclic encoding        |
| pandas             | >=2.0.0   | Data loading, feature engineering            |
| scikit-learn       | >=1.3.0   | Models, pipelines, evaluation metrics        |
| imbalanced-learn   | >=0.11.0  | SMOTE oversampling for class imbalance       |
| joblib             | >=1.3.0   | Saving / loading the trained model (.pkl)    |
| shap               | >=0.44.0  | Model explainability (SHAP values)           |
| matplotlib         | >=3.7.0   | All plots & figure generation                |
| seaborn            | >=0.13.0  | Confusion matrix heatmaps                    |

---

## Python Version

Python **3.9 or higher** is recommended. SHAP and imbalanced-learn both
require Python 3.8+ and work best with 3.9–3.12.
