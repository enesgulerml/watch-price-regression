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


class DataImputation:
    """
    Pipeline Step 3: Imputation.
    - Loads config.yaml.
    - Reads the output of Step 2.
    - Applies the strategies defined in 'config.imputation_rules'.
    - Includes a fallback strategy to fix error #3 (Sneaky NaN).
    - Saves the result to 'data/processed/'.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Data Imputation (Step 3) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.rules = self.config.imputation_rules
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Loads the data from step 2 (outlier removal)."""
        input_path = PROJECT_ROOT / self.paths.step_02_outliers_removed
        logger.info(f"Loading step 2 data: {input_path}")
        try:
            return pd.read_pickle(input_path)
        except FileNotFoundError:
            logger.error(f"FATAL: Step 2 data not found: {input_path}. Run Step 2 first.")
            raise

    def apply_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies padding rules from config."""
        processed_df = df.copy()

        # Strategy 1: Fixed Value Filling
        logger.info("Strategy 1: Applying constant value padding...")
        for col, value in self.rules.fixed_value.items():
            processed_df[col].fillna(value, inplace=True)

        # Strategy 2: Contextual (Groupby Median) Filling
        logger.info("Strategy 2: Applying contextual (groupby median) padding...")
        for rule in self.rules.groupby_median:
            target_col = rule.target_col
            groupby_cols = rule.groupby_cols
            logger.info(f"Filling: '{target_col}' (Group: {groupby_cols})")

            processed_df[target_col].fillna(
                processed_df.groupby(groupby_cols)[target_col].transform('median'),
                inplace=True
            )

        # Strategy 3: FALLBACK
        # If there is STILL NaN left after Groupby (Meteorite example),
        # fill them with the global median/mode.
        logger.info("Strategy 3: Applying Fallback (Sneaky NaN correction)...")

        # Let's just get our numeric columns (from config)
        num_cols = self.config.columns.numerical_features
        for col in num_cols:
            if processed_df[col].isnull().any():
                logger.warning(f"FALLBACK: Still found NaN in column '{col}'. Filling with global median.")
                global_median = processed_df[col].median()
                processed_df[col].fillna(global_median, inplace=True)

        # Let's also check the categorical columns (our fixed_value should have taken care of all of them though)
        cat_cols = self.config.columns.categorical_features
        for col in cat_cols:
            if processed_df[col].isnull().any():
                logger.warning(f"FALLBACK: Still found NaN in column '{col}'. Filling with global mode ('most frequent').")
                global_mode = processed_df[col].mode()[0]
                processed_df[col].fillna(global_mode, inplace=True)

        logger.info("Missing data filling completed.")
        return processed_df

    def save_data(self, df: pd.DataFrame):
        """Saves the processed data for the next step."""
        output_path = PROJECT_ROOT / self.paths.step_03_imputed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_pickle(output_path)
        logger.info(f"The filled data (Step 3) is saved to: {output_path}")

    def run(self):
        """Runs the entire flow of Step 3."""
        try:
            data = self.load_data()
            imputed_data = self.apply_imputation(data)
            self.save_data(imputed_data)
            logger.info("Data Imputation (Step 3) completed successfully.")
        except Exception as e:
            logger.error(f"An error occurred during Data Imputation (Step 3): {e}")
            raise


if __name__ == '__main__':
    imputation_pipeline = DataImputation()
    imputation_pipeline.run()