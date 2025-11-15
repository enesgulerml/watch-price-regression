import pytest
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- Project Path Setup (Fixing Captain's Error #31) ---
# This test file is in 'tests/', so the ROOT is 1 level up.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / 'config/config.yaml'
MODELS_PATH = PROJECT_ROOT / 'models'

# We must load the config *without* the 'src.utils' helper,
# as this test should be isolated.
import yaml


def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        pytest.fail(f"FATAL: Config file not found at {CONFIG_PATH}")
    except Exception as e:
        pytest.fail(f"FATAL: Error loading config file: {e}")


# --- Test Fixtures (Reusable Test Data) ---

@pytest.fixture(scope="module")
def config():
    """Load the main config file once per test module."""
    return load_config()


@pytest.fixture(scope="module")
def preprocessor(config):
    """Load the v2 preprocessor (scaler) once per test module."""
    preprocessor_path = PROJECT_ROOT / config['paths']['scaler']
    try:
        return joblib.load(preprocessor_path)
    except FileNotFoundError:
        pytest.fail(f"FATAL: Preprocessor not found at {preprocessor_path}. Run the v1 pipeline first.")


@pytest.fixture(scope="module")
def model(config):
    """Load the v2 champion model once per test module."""
    model_path = PROJECT_ROOT / config['paths']['model']
    try:
        return joblib.load(model_path)
    except FileNotFoundError:
        pytest.fail(f"FATAL: Model not found at {model_path}. Run the v1 pipeline first.")


@pytest.fixture
def sample_watch_data(config):
    """
    Create a single, valid sample watch input as a DataFrame.
    This MUST match the schema our preprocessor expects.
    """
    # Use the same schema from app/schema.py example
    data = {
        "Case Diameter": [40.5],
        "Water Resistance": [10],
        "Warranty (Years)": [2],
        "Weight (g)": [85.0],
        "Brand": ["Seiko"],
        "Gender": ["Male"],
        "Case Color": ["Silver"],
        "Glass Shape": ["Flat"],
        "Origin": ["Japan"],
        "Case Material": ["Steel"],
        "Additional Feature": ["Luminous"],
        "Strap Color": ["Black"],
        "Strap Material": ["Leather"],
        "Mechanism": ["Automatic"],
        "Glass Type": ["Sapphire"],
        "Dial Color": ["Blue"]
    }

    # Ensure column order matches the config
    num_cols = config['columns']['numerical_features']
    cat_cols = config['columns']['categorical_features']

    df = pd.DataFrame(data, index=[0])
    return df[num_cols + cat_cols]


# --- Tests ---

def test_pipeline_integration(preprocessor, model, sample_watch_data):
    """
    The main integration test (v6).
    Checks if the preprocessor and model can work together.
    """
    # 1. Test the Preprocessor
    try:
        transformed_data = preprocessor.transform(sample_watch_data)
    except Exception as e:
        pytest.fail(f"Preprocessor (scaler.joblib) failed to transform sample data: {e}")

    assert transformed_data is not None
    assert transformed_data.shape[0] == 1  # Should be one row

    # 2. Test the Model
    try:
        prediction_log = model.predict(transformed_data)
    except Exception as e:
        pytest.fail(f"Model (model.joblib) failed to predict on transformed data: {e}")

    # 3. Test the Output
    assert prediction_log is not None
    assert isinstance(prediction_log[0], (np.float32, np.float64, float))  # Must be a float number

    # 4. Test the Value (Sanity Check)
    # Inverse transform to USD
    prediction_usd = np.expm1(prediction_log[0])

    # A $500k watch or a $0 watch would be an error
    assert 100 < prediction_usd < 100000