# End-to-End Watch Price Regression (MLOps Pipeline v1)

This is an end-to-end Machine Learning Operations (MLOps) pipeline built to "Google-level" standards, designed to predict the price of a watch based on its features.

This repository demonstrates the process of refactoring a "v0" prototype—which suffered from critical flaws like **Data Leakage** and the **Curse of Dimensionality**—into a robust, reproducible, and configuration-driven 6-step pipeline.

## 🏆 v1 Success Metrics (XGBoost Baseline)

After identifying and fixing 5 "Cardinal Sins" (critical bugs) from the prototype, this v1 pipeline achieves the following **honest, leak-free** results:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R2)** | `0.979` | The model successfully explains 97.9% of the variance in (log-transformed) price. |
| **Mean Absolute Error (MAE)** | `$572.25` | In a market where prices range from $500 to $51,000+, our model's predictions are, on average, only **$572** off the actual price. |

---

## 🏛️ MLOps Architecture: The Brain & The Muscle

This project is built on the core MLOps principle of **Separating Logic from Configuration**.

* **`config/config.yaml` (The Brain):**
  This file is the "single source of truth." All file paths, "magic numbers" (e.g., `max: 40mm` for female watches), outlier rules, imputation strategies, and model parameters are defined here. This allows us to change the entire experiment without touching the source code.

* **`src/` (The Muscle):**
  This directory contains "dumb" but powerful Python scripts. These scripts do not know *what* to do; they only know how to read instructions from the "Brain" (`config.yaml`) and execute them.

---

## ⛓️ The v1 Pipeline (A 6-Step Walkthrough)

The core logic lives in `src/pipeline/` as a series of modular, interdependent scripts that feed their output to the next step.

1.  **`_01_data_ingestion.py`**:
    * Loads the raw data.
    * **Fixes Cardinal Sin #1:** Drops rows with a missing target variable (`Price is NaN`).
    * Drops duplicate records.
2.  **`_02_outlier_removal.py`**:
    * Reads domain-knowledge-based rules from the "Brain" (e.g., `Case Diameter` rules based on `Gender`).
3.  **`_03_imputation.py`**:
    * Reads multi-strategy (`fixed-value`, `groupby-median`) rules from the "Brain".
    * **Fixes Cardinal Sin #3:** Includes a robust "Fallback" strategy to catch and fix "sneaky NaNs" (like the `Glass Type` warning we saw).
4.  **`_04_feature_creation.py`**:
    * Applies `Log-Transform` to the target variable (`Price (USD)` -> `Log_Price_USD`).
    * **Fixes Cardinal Sins #4 & #5:** Drops the Data Leakage feature (`Price_Segment`) and the Curse of Dimensionality feature (`Model`).
5.  **`_05_data_transformation.py`**:
    * Splits data into Train/Test sets (preventing leakage).
    * Uses a `ColumnTransformer` (Scaler for numeric, OneHotEncoder for categorical) to prepare data for the model.
    * Saves the final, fitted preprocessor (`scaler.joblib`) for inference.
6.  **`_06_model_training.py`**:
    * Loads the final, clean data.
    * Trains the baseline XGBoost model.
    * Evaluates metrics in both Log-space (R2) and Dollar-space (MAE).
    * Saves the final model (`model.joblib`).

---

## 🚀 How to Run (Setup)

This project is managed using `conda`.

### 1. Clone the Repository

```bash
git clone [https://github.com/enesml/watch-price-regression.git](https://github.com/enesml/watch-price-regression.git)
cd watch-price-regression
```

### 2. Create the Conda Environment
All dependencies are specified in environment.yml.

```bash
conda env create -f environment.yml
conda activate watch-ml
```

### 3. Download the Dataset
The dataset is hosted on Kaggle.

Download the dataset [from here](https://www.kaggle.com/datasets/enesml/artificial-watch-price-prediction-dataset)

After downloading, copy the .csv file into the data/raw/ directory. Ensure its name is watch_price_raw.csv (or update the path in config/config.yaml).

### 4. Run the Full v1 Pipeline
You can run the pipeline step-by-step (for debugging) or as a full sequence.

```bash
# Run each step in order
python src/pipeline/_01_data_ingestion.py
python src/pipeline/_02_outlier_removal.py
python src/pipeline/_03_imputation.py
python src/pipeline/_04_feature_creation.py
python src/pipeline/_05_data_transformation.py
python src/pipeline/_06_model_training.py
```

