import pandas as pd
from pathlib import Path
import logging
import sys
import numpy as np

# Add the project root directory to the Python path

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


class FeatureCreation:
    """
    Pipeline Step 4: Feature Engineering (Basic).
    - Fixes Bug #4 (Data Leakage) and Bug #5 (Dimensionality Curse).
    - Applies Log-Transform to the target variable (Price).
    - Discards unnecessary and noisy columns (defined in the config).
    - Saves the result to 'data/processed/'.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Feature Creation (Step 4) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.cols = self.config.columns
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Loads the data from step 3 (imputation)."""
        input_path = PROJECT_ROOT / self.paths.step_03_imputed
        logger.info(f"Loading step 3 data: {input_path}")
        try:
            return pd.read_pickle(input_path)
        except FileNotFoundError:
            logger.error(f"FATAL: Step 3 data not found: {input_path}. Run Step 3 first.")
            raise

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies feature engineering and cleaning steps."""
        processed_df = df.copy()

        # 1. Log-Transform
        logger.info(f"Applying log-transform to target variable '{self.cols.target_original}'.")
        processed_df[self.cols.target_log_transformed] = np.log1p(processed_df[self.cols.target_original])

        # 2. Remove Unnecessary Columns
        cols_to_drop = [self.cols.target_original] + self.cols.cols_to_drop

        # Discard what actually exists in the DataFrame
        cols_to_drop_existing = [col for col in cols_to_drop if col in processed_df.columns]

        processed_df.drop(columns=cols_to_drop_existing, inplace=True)
        logger.info(f"Unnecessary/Noisy columns discarded: {cols_to_drop_existing}")

        return processed_df

    def save_data(self, df: pd.DataFrame):
        """Saves the processed data for the next step."""
        output_path = PROJECT_ROOT / self.paths.step_04_features_created
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_pickle(output_path)
        logger.info(f"The characterized data (Step 4) was saved to: {output_path}")

    def run(self):
        """Runs the entire flow of Step 4."""
        try:
            data = self.load_data()
            featured_data = self.create_features(data)
            self.save_data(featured_data)
            logger.info("Feature Creation (Step 4) completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during Feature Creation (Step 4): {e}")
            raise


if __name__ == '__main__':
    feature_pipeline = FeatureCreation()
    feature_pipeline.run()