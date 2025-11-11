import pandas as pd
from pathlib import Path
import logging
import sys
import joblib
import numpy as np
import optuna
import mlflow
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- MLFlow Installation ---
MLFLOW_TRACKING_URI = "file:./mlruns"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Watch Price - XGBoost Tuning (v2)")


def load_final_data(config):
    """Loads the final datasets (from Step 5)."""
    logger.info("Loading final datasets (data/final/)...")
    try:
        X_train = joblib.load(PROJECT_ROOT / config.paths.X_train)
        y_train = joblib.load(PROJECT_ROOT / config.paths.y_train)
        X_test = joblib.load(PROJECT_ROOT / config.paths.X_test)
        y_test = joblib.load(PROJECT_ROOT / config.paths.y_test)
        return X_train, y_train, X_test, y_test
    except FileNotFoundError:
        logger.error(f"FATAL: Final data sets not found. Run Step 5 first.")
        raise


def objective(trial: optuna.Trial, X_train, y_train, X_test, y_test):
    """
    The function that Optuna will run for each 'trial.'
    Purpose: Minimize MAE (error).
    """

    # 1. START the experiment in MLFlow
    with mlflow.start_run():
        # 2. Request "experiment parameters" from Optuna (Search Field)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        # 3. Save parameters to MLFlow
        mlflow.log_params(params)

        # 4. Train the model with these parameters
        model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            **params # Parameters provided by Optuna
        )
        model.fit(X_train, y_train)

        # 5. Evaluate Model
        y_pred_log = model.predict(X_test)

        # Return to original Dollar ($) field
        y_true_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred_log)

        mae_usd = mean_absolute_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_test, y_pred_log)

        # 6. Save Metrics (Results) to MLFlow
        mlflow.log_metric("mae_usd", mae_usd)
        mlflow.log_metric("r2_log", r2)

    # 7. Return to Optuna the value it should "minimize" (MAE)
    return mae_usd

    # 2. Start a new record (run) for this "experiment" in MLFlow
    with mlflow.start_run(nested=True):

        # 3. Save parameters to MLFlow
        mlflow.log_params(params)

        # 4. Train the model with these parameters
        model = XGBRegressor(
            random_state=42,
            n_jobs=-1,
            **params  # Parameters provided by Optuna
        )
        model.fit(X_train, y_train)

        # 5. Evaluate Model
        y_pred_log = model.predict(X_test)

        # Return to original Dollar ($) field
        y_true_orig = np.expm1(y_test)
        y_pred_orig = np.expm1(y_pred_log)

        mae_usd = mean_absolute_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_test, y_pred_log)

        # 6. Save Metrics (Results) to MLFlow
        mlflow.log_metric("mae_usd", mae_usd)
        mlflow.log_metric("r2_log", r2)

    # 7. Return to Optuna the value it should "minimize" (MAE)
    return mae_usd


def main():
    """Runs the main tuning flow."""

    config = load_config()
    X_train, y_train, X_test, y_test = load_final_data(config)

    # We say to Optuna: "Try to reduce MAE (direction='minimize')"
    study = optuna.create_study(direction='minimize')


    # We tell Optuna: "Run the objective function 50 times (n_trials=50)"
    logger.info("Optimization (Hyperparameter Tuning) is starting... (50 Experiments)")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_test, y_test),
        n_trials=50,
        show_progress_bar=True
    )

    # When 50 experiment is over
    logger.info("Optimization completed.")
    logger.info(f"Best MAE (Error): ${study.best_value:,.2f}")
    logger.info("Best Params:")
    logger.info(study.best_params)

    logger.info("\n--- Initializing the MLFlow Interface ---")
    logger.info("To see the experiments, open a NEW terminal and run the following command:")
    logger.info("mlflow ui --port 5001")
    logger.info("Then open http://127.0.0.1:5001 in your browser.")


if __name__ == '__main__':
    main()