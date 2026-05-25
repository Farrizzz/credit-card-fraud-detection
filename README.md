# redit Card Fraud Detection — EDA & Baseline Model

> **Project 1 dari 3** dalam portfolio Data Analyst saya | Domain: Fintech

---

## Overview

Proyek ini menganalisis dataset transaksi kartu kredit untuk mendeteksi pola fraud menggunakan Exploratory Data Analysis (EDA), SQL queries, dan Machine Learning. Mencakup tantangan nyata di industri fintech: **class imbalance** yang ekstrem (~0.17% fraud).

## Tujuan

1. Memahami distribusi dan pola transaksi fraud vs normal
2. Mengidentifikasi fitur-fitur paling diskriminatif
3. Membangun baseline model klasifikasi yang andal
4. Mengevaluasi model dengan metrik yang tepat (bukan hanya accuracy)

---

## Struktur Project

```
project1_fraud/
│
├── 01_exploratory_analysis.ipynb   # EDA lengkap: distribusi, temporal, korelasi
├── 02_sql_queries.ipynb            # Analisis ulang dengan DuckDB SQL
├── 03_baseline_model.ipynb         # ML: Logistic Regression + Random Forest
│
├── plot_01_class_imbalance.png
├── plot_02_fraud_by_hour.png
├── plot_03_amount_analysis.png
├── plot_04_correlation.png
├── plot_05_feature_distributions.png
├── plot_sql_01_hourly.png
├── plot_sql_02_amount_bracket.png
├── plot_sql_03_high_value_fraud.png
├── plot_model_01_smote.png
├── plot_model_02_evaluation.png
├── plot_model_03_feature_importance.png
├── plot_model_04_comparison.png
│
└── README.md
```

---

## 📊 Dataset

- **Sumber:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Jumlah transaksi:** 284,807
- **Fitur:** V1–V28 (hasil PCA), Time, Amount, Class
- **Label:** 0 = Normal, 1 = Fraud

---

## Key Findings

### EDA
| Insight | Detail |
|---------|--------|
| Class imbalance | Hanya 492 fraud dari 284,807 transaksi (0.172%) |
| Pola waktu | Fraud rate tertinggi di dini hari (jam 2–4 pagi) |
| Nilai transaksi | Median fraud ($22) lebih rendah dari normal ($22.7) tapi ada outlier besar |
| Fitur terpenting | V4, V11, V14, V17 paling berkorelasi dengan label fraud |

### Model Performance
| Model | AUC-ROC | F1-Score | Precision | Recall |
|-------|---------|----------|-----------|--------|
| Logistic Regression | ~0.97 | ~0.74 | ~0.89 | ~0.63 |
| **Random Forest** | **~0.98** | **~0.85** | **~0.96** | **~0.77** |

>  Random Forest unggul di semua metrik — dipilih sebagai model final.

---

## Tech Stack

- **Python:** pandas, numpy, matplotlib, seaborn, scipy
- **SQL:** DuckDB (query langsung dari DataFrame)
- **Machine Learning:** scikit-learn, imbalanced-learn (SMOTE)
- **Teknik:** Feature Engineering, SMOTE Oversampling, Stratified Split

---

## Cara Menjalankan

```bash
# 1. Clone repo
git clone https://github.com/username/project1-fraud-detection.git
cd project1-fraud-detection

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn duckdb jupyter

# 3. Download dataset
# https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Simpan creditcard.csv di folder ini

# 4. Jalankan notebook secara berurutan
jupyter notebook 01_exploratory_analysis.ipynb
```

---

## Next Steps

Temuan dari project ini akan digunakan di **Project 3: Loan Default Prediction** dengan:
- XGBoost + Hyperparameter Tuning
- SHAP values untuk model interpretability
- Business recommendation report

---

*Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)*
