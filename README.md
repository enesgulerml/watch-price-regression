# End-to-End Watch Price Regression (MLOps v2: Optimized)

This is an end-to-end Machine Learning Operations (MLOps) pipeline built to "Google-level" standards, designed to predict the price of a watch based on its features.

This repository demonstrates a full MLOps lifecycle:
1.  **v1 (Baseline):** A robust 6-step pipeline was built to fix a flawed prototype, achieving a baseline MAE of `$572`.
2.  **v2 (Optimization):** The baseline was "promoted" to v2 by using **MLFlow** and **Optuna** to run 50 hyperparameter tuning experiments, finding a champion model that **reduced the MAE to $514.**

## 🏆 v2 Success Metrics (XGBoost Champion)

The final v2 model, found via MLFlow/Optuna and saved in `models/model.joblib`, achieves the following honest, leak-free results:

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **R-squared (R2)** | `0.98+` | The model successfully explains over 98% of the variance. |
| **Mean Absolute Error (MAE)** | **`$514.33`** | (v1 Baseline was $572.25). Our v2 model is, on average, only **$514** off the actual price. |

---

## 🏛️ MLOps Architecture: The Brain & The Muscle

This project is built on the core MLOps principle of **Separating Logic from Configuration**.

* **`config/config.yaml` (The Brain):**
  This file is the "single source of truth." All file paths, outlier rules, imputation strategies, and **final champion model parameters** (found by Optuna) are defined here.

* **`src/` (The Muscle):**
  This directory contains the Python scripts that execute the pipeline. These scripts read their instructions *dynamically* from the "Brain."

---

## ⛓️ The MLOps Workflow

### v1: The 6-Step Pipeline
The core pipeline (`src/pipeline/`) fixes 5 "Cardinal Sins" (Data Leakage, etc.) from the prototype.
1.  `_01_data_ingestion.py`
2.  `_02_outlier_removal.py`
3.  `_03_imputation.py`
4.  `_04_feature_creation.py`
5.  `_05_data_transformation.py`
6.  `_06_model_training.py` (Trains the final model using params from `config.yaml`)

### v2: Experimentation & Optimization (The "Lab")
* **`src/run_tuning.py`**: This script uses **MLFlow** (the lab notebook) and **Optuna** (the scientist) to run 50 experiments to find the best hyperparameters.
* **`mlruns/`**: The MLFlow UI database, proving our experimentation work.

---

## 🚀 How to Run (Setup)

This project is managed using `conda`.

### 1. Clone the Repository
```bash
git clone [https://github.com/enesml/watch-price-regression.git](https://github.com/enesml/watch-price-regression.git)
cd watch-price-regression
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
conda activate watch-ml
```

### 3. Download the Dataset

* Download the dataset [from here](https://www.kaggle.com/datasets/enesml/artificial-watch-price-prediction-dataset)
* Copy the .csv file into data/raw/ and ensure its name is watch_price_raw.csv (or update config.yaml).


### 4. Run the Full v2 Pipeline (Train the Champion)
This runs the 6-step pipeline, which now automatically uses the "champion" parameters from config.yaml.

```bash
python src/pipeline/_01_data_ingestion.py
python src/pipeline/_02_outlier_removal.py
# ... run all 6 steps ...
python src/pipeline/_06_model_training.py
```

### 5. (Optional) Run Your Own Experiments
To run your own 50-trial optimization (and see MLFlow in action):

```bash
# In Terminal 1:
mlflow ui --port 5001

# In Terminal 2:
python src/run_tuning.py
```