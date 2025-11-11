import pandas as pd
from pathlib import Path
import logging
import sys

# Add the project root directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

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


class OutlierRemoval:
    """
    Pipeline Step 2: Domain-Based Outlier Cleaning.
    - Loads config.yaml.
    - Reads the output of Step 1.
    - Dynamically applies the rules defined in 'config.outlier_rules'.
    - Saves the result to 'data/processed/'.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Outlier Removal (Step 2) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.rules = self.config.outlier_rules
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_cleaned_data(self) -> pd.DataFrame:
        """Loads the data from step 1 (ingestion)."""
        input_path = PROJECT_ROOT / self.paths.step_01_cleaned
        logger.info(f"Loading step 1 data: {input_path}")
        try:
            return pd.read_pickle(input_path)
        except FileNotFoundError:
            logger.error(f"FATAL: Step 1 data not found: {input_path}. Run Step 1 first.")
            raise

    def apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dynamically applies rules from config."""
        processed_df = df.copy()
        initial_rows = len(processed_df)

        # Return in 'outlier_rules' in config.yaml
        for col, gender_rules in self.rules.items():
            logger.info(f"Aykırı değer kuralı uygulanıyor: {col}")

            # Return for all genders (Female, Male, Unisex)
            for gender, limits in gender_rules.items():
                min_val = limits.get('min', -float('inf'))
                max_val = limits.get('max', float('inf'))

                # Define the rule:
                # (Gender == 'Female' AND (Col < min OR Col > max))
                condition = (
                        (processed_df['Gender'] == gender) &
                        ((processed_df[col] < min_val) | (processed_df[col] > max_val))
                )

                # Keep lines that DO NOT meet this rule (~ condition)
                processed_df = processed_df[~condition]

        rows_dropped = initial_rows - len(processed_df)
        if rows_dropped > 0:
            logger.info(f"{rows_dropped} rows that were caught in the outlier rule were dropped.")
        else:
            logger.info("No outliers found or all data conforms to the rules.")

        return processed_df

    def save_processed_data(self, df: pd.DataFrame):
        """Saves the processed data for the next step."""
        output_path = PROJECT_ROOT / self.paths.step_02_outliers_removed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_pickle(output_path)
        logger.info(f"The outlier-cleaned data (Step 2) was saved to: {output_path}")

    def run(self):
        """Runs the entire flow of Step 2."""
        try:
            cleaned_df = self.load_cleaned_data()
            processed_df = self.apply_rules(cleaned_df)
            self.save_processed_data(processed_df)
            logger.info("Outlier Removal (Step 2) completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during Outlier Removal (Step 2): {e}")
            raise


if __name__ == '__main__':
    outlier_pipeline = OutlierRemoval()
    outlier_pipeline.run()