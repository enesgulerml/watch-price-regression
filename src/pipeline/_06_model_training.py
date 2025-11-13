import pandas as pd
from pathlib import Path
import logging
import sys
import joblib
import numpy as np

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Add the project root directory to the Python path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Import the 'utils' module under 'src'
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


class ModelTraining:
    """
    The sixth and final step of the pipeline: Model Training and Evaluation.
    - Loads the final data (from Step 5).
    - Trains an XGBoost model.
    - Evaluates the model in both the 'log' and 'original' (dollar) domains.
    - Saves the winning model to 'models/'.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Model Training (Step 6) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.params = self.config.training_params
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_data(self):
        """Loads final datasets."""
        logger.info("Loading final datasets (data/final/)...")
        try:
            X_train = joblib.load(PROJECT_ROOT / self.paths.X_train)
            y_train = joblib.load(PROJECT_ROOT / self.paths.y_train)
            X_test = joblib.load(PROJECT_ROOT / self.paths.X_test)
            y_test = joblib.load(PROJECT_ROOT / self.paths.y_test)
            return X_train, y_train, X_test, y_test
        except FileNotFoundError:
            logger.error(f"FATAL: Final data sets not found. Run Step 5 first.")
            raise

    def evaluate_model(self, y_true_log, y_pred_log):
        """Calculates metrics at both log and original prices."""

        # 1. Metrics in logarithmic space (what the model sees)
        r2 = r2_score(y_true_log, y_pred_log)
        rmse_log = np.sqrt(mean_squared_error(y_true_log, y_pred_log))

        logger.info("--- Logarithmic Area Metrics (As Seen by the Model) ---")
        logger.info(f"  R-squared (R2): {r2:.4f}")
        logger.info(f"  RMSE (Log Price): {rmse_log:.4f}")

        # 2. Metrics in the Original Price field (What we see - Dollars $)
        y_true_orig = np.expm1(y_true_log)
        y_pred_orig = np.expm1(y_pred_log)

        mae_orig = mean_absolute_error(y_true_orig, y_pred_orig)
        rmse_orig = np.sqrt(mean_squared_error(y_true_orig, y_pred_orig))

        logger.info("--- Original Price Area Metrics (USD $) ---")
        logger.info(f"  Mean Absolute Error (MAE): ${mae_orig:,.2f} (Mean Error)")
        logger.info(f"  Root Mean Squared Error (RMSE): ${rmse_orig:,.2f}")

    def run(self):
        """Runs the entire flow of Step 6."""
        try:
            X_train, y_train, X_test, y_test = self.load_data()

            # 1. Define Model (with v2 Champion Parameters from config.yaml)
            champion_params = self.config.training_params.xgboost_params
            model = XGBRegressor(
                random_state=self.params.random_state,
                n_jobs=-1,
                **champion_params  # "Champion" parameters from Optuna
            )

            # 2. Train Model
            logger.info("Training XGBoost model...")
            model.fit(X_train, y_train)
            logger.info("Model training completed.")

            # 3. Save Model
            model_path = PROJECT_ROOT / self.paths.model
            joblib.dump(model, model_path)
            logger.info(f"The trained model is saved to: {model_path}")

            # 4. Evaluate the Model (with test data)
            logger.info("Evaluating the model on test data...")
            y_pred_log = model.predict(X_test)

            self.evaluate_model(y_test, y_pred_log)

            logger.info("Model Training (Step 6) completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during Model Training (Step 6): {e}")
            raise


if __name__ == '__main__':
    training_pipeline = ModelTraining()
    training_pipeline.run()