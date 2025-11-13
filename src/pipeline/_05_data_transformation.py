import pandas as pd
from pathlib import Path
import logging
import sys
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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


class DataTransformation:
    """
    Step 5 of the Pipeline: Data Transformation (Preparing the Model).
    - Splits the data into x (feature) and y (target).
    - Splits into Train/Test.
    - Creates a ColumnTransformer (preprocessor) for the categorical and numeric columns.
    - Fits the preprocessor to the train data and transforms the test data.
    - Saves the final datasets and the preprocessor.
    """

    def __init__(self, config_path=Path('config/config.yaml')):
        logger.info("Data Transformation (Step 5) begins...")
        try:
            self.config = load_config(config_path)
            self.paths = self.config.paths
            self.cols = self.config.columns
            self.params = self.config.training_params
        except Exception as e:
            logger.error(f"ERROR while loading configuration: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Loads data from step 4 (feature creation)."""
        input_path = PROJECT_ROOT / self.paths.step_04_features_created
        logger.info(f"Loading step 4 data: {input_path}")
        try:
            return pd.read_pickle(input_path)
        except FileNotFoundError:
            logger.error(f"FATAL: Step 4 data not found: {input_path}. Run Step 4 first.")
            raise

    def build_preprocessor(self) -> ColumnTransformer:
        """Creates a ColumnTransformer (scaler + encoder) according to the config."""

        numeric_features = self.cols.numerical_features
        categorical_features = self.cols.categorical_features

        # Digital converter (pipeline)
        numeric_transformer = StandardScaler()

        # Categorical converter (pipeline)
        # handle_unknown='ignore' means to throw an error if a category that has NOT been seen in the test set appears
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        # Combine the two with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='passthrough'# Do not touch the remaining columns (if any)
        )
        logger.info("ColumnTransformer (Preprocessor) created successfully.")
        return preprocessor

    def run(self):
        """Runs the entire flow of Step 5."""
        try:
            # 1. Load Data
            df = self.load_data()

            # 2. Separate x and y
            logger.info(f"Data is split into X (feature) and y (target). Target: '{self.cols.target_log_transformed}'")
            X = df.drop(columns=[self.cols.target_log_transformed])
            y = df[self.cols.target_log_transformed]

            # 3. Train/Test Split
            logger.info("Data is split into Train/Test...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.params.test_size,
                random_state=self.params.random_state
            )
            logger.info(f"Split tamamlandı. Train: {len(X_train)}, Test: {len(X_test)}")

            # 4. Create Preprocessor
            preprocessor = self.build_preprocessor()

            # 5. Apply Preprocessor (FIT -> Train, TRANSFORM -> Test)
            logger.info("Preprocessor (Scaler/Encoder) X_train verisine 'fit_transform' ediliyor...")
            X_train_transformed = preprocessor.fit_transform(X_train)

            logger.info("Preprocessor 'transforming' data to X_test...")
            X_test_transformed = preprocessor.transform(X_test)

            # 6. Save Preprocessor (Entire Pipeline)
            preprocessor_path = PROJECT_ROOT / self.paths.scaler
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor (Scaler+Encoder) şuraya kaydedildi: {preprocessor_path}")

            # 7. Save Final Datasets
            (PROJECT_ROOT / self.paths.X_train).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(X_train_transformed, PROJECT_ROOT / self.paths.X_train)
            joblib.dump(y_train, PROJECT_ROOT / self.paths.y_train)
            joblib.dump(X_test_transformed, PROJECT_ROOT / self.paths.X_test)
            joblib.dump(y_test, PROJECT_ROOT / self.paths.y_test)

            logger.info(f"Final (X_train, y_train, X_test, y_test) sets are saved in the 'data/final/' folder.")
            logger.info("Data Transformation (Step 5) completed successfully.")

        except Exception as e:
            logger.error(f"An error occurred during Data Transformation (Step 5): {e}")
            raise


if __name__ == '__main__':
    transformation_pipeline = DataTransformation()
    transformation_pipeline.run()