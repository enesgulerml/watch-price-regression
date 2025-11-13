import pandas as pd
from pathlib import Path
import logging
import sys

# Add the project root directory (watch-price-regression) to the Python path.

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Now we can import the 'utils' module under 'src'
from src.utils import load_config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)  # Print logs to terminal
    ]
)
logger = logging.getLogger(__name__)


class DataIngestion:
    """
    Pipeline Step 1: Data Loading and Basic Cleanup.
    - Loads config.yaml.
    - Reads raw data.
    - Fixes bug #1: Discards rows with missing target (Price).
    - Discards duplicate rows.
    - Saves the result to 'data/processed/' for the next step.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Data Ingestion (Step 1) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.target_col = self.config.columns.target_original
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_raw_data(self) -> pd.DataFrame:
        """Loads raw data from the path specified in config."""
        raw_data_path = PROJECT_ROOT / self.paths.raw_data
        logger.info(f"Loading raw data: {raw_data_path}")
        try:
            return pd.read_csv(raw_data_path)
        except FileNotFoundError:
            logger.error(f"FATAL: Raw data file not found: {raw_data_path}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies basic cleaning steps."""
        processed_df = df.copy()

        # 1. Skip duplicate lines
        initial_rows = len(processed_df)
        processed_df.drop_duplicates(inplace=True)
        rows_dropped = initial_rows - len(processed_df)
        if rows_dropped > 0:
            logger.info(f"{rows_dropped} duplicate rows were dropped.")

        # 2. Discard Missing Target Variable
        initial_rows = len(processed_df)
        processed_df.dropna(subset=[self.target_col], inplace=True)
        rows_dropped = initial_rows - len(processed_df)
        if rows_dropped > 0:
            logger.info(f"{rows_dropped} rows without '{self.target_col}' (Price) information were dropped.")

        return processed_df

    def save_cleaned_data(self, df: pd.DataFrame):
        """Saves the cleaned data for the next step."""
        output_path = PROJECT_ROOT / self.paths.step_01_cleaned
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_pickle(output_path)
        logger.info(f"Cleaned data (Step 1) saved to: {output_path}")

    def run(self):
        """Runs the entire flow of Step 1."""
        try:
            raw_df = self.load_raw_data()
            cleaned_df = self.clean_data(raw_df)
            self.save_cleaned_data(cleaned_df)
            logger.info("Data Ingestion (Step 1) completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during Data Ingestion (Step 1): {e}")
            raise


if __name__ == '__main__':
    ingestion_pipeline = DataIngestion()
    ingestion_pipeline.run()