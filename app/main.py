import pandas as pd
import joblib
import numpy as np
import uvicorn
import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

# --- Clean Project Path Setup ---
# 'app/main.py' does NOT need the 'sys.path' HACK to find 'src' or 'app'.
# Why? Because 'pip install -e .' already registered 'src' and 'app'
# as installable packages in our 'conda' environment.
from app.schema import WatchFeatures
from src.utils import load_config

# However, we DO need to know the Project Root to find non-code
# assets like 'config/' and 'models/'.
# We define this as a global variable, outside all functions.
# (app/main.py -> app/ -> watch-price-regression/ (ROOT))
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model & Config Loading (Once, at Startup) ---
# We load these globally.
# 'lifespan' will fill the 'models' dict.
# 'config' will be accessible everywhere.
models = {}
config = load_config(PROJECT_ROOT / 'config/config.yaml')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 'lifespan' event: Runs BEFORE the server starts.
    Loads the models ONCE and stores them in the global 'models' dict.
    """
    try:
        # 1. Load the "Key" (Preprocessor)
        preprocessor_path = PROJECT_ROOT / config.paths.scaler
        models["preprocessor"] = joblib.load(preprocessor_path)

        # 2. Load the "Champion Engine" (v2 Model)
        model_path = PROJECT_ROOT / config.paths.model
        models["model"] = joblib.load(model_path)

        logger.info(f"Models loaded successfully. (Preprocessor and Model are in memory)")

    except Exception as e:
        logger.error(f"FATAL ERROR during model loading: {e}")
        raise

    yield  # The server is now "running"

    # When the server shuts down...
    models.clear()
    logger.info("Models cleared from memory.")


# Initialize the FastAPI app with our 'lifespan' event
app = FastAPI(
    lifespan=lifespan,
    title="Watch Price Prediction API (v2)",
    description="MLOps v2 (Optuna-tuned XGBoost) model serving API.",
    version="2.0.0"
)


@app.get("/", tags=["Health Check"])
def health_check():
    """Check if the server is alive."""
    return {"status": "ok", "message": "API is running!"}


@app.post("/predict", tags=["Prediction"])
def predict(features: WatchFeatures):
    """
    Receive watch features (JSON) and return a price prediction (JSON).
    """
    try:
        # 1. Convert Pydantic schema to a dictionary
        input_data = features.model_dump()

        # 2. Translate: Pydantic names -> DataFrame column names
        # This creates the exact input format our preprocessor expects.
        data_for_df = {
            'Case Diameter': input_data['Case_Diameter'],
            'Water Resistance': input_data['Water_Resistance'],
            'Warranty (Years)': input_data['Warranty_Years'],
            'Weight (g)': input_data['Weight_g'],
            'Brand': input_data['Brand'],
            'Gender': input_data['Gender'],
            'Case Color': input_data['Case_Color'],
            'Glass Shape': input_data['Glass_Shape'],
            'Origin': input_data['Origin'],
            'Case Material': input_data['Case_Material'],
            'Additional Feature': input_data['Additional_Feature'],
            'Strap Color': input_data['Strap_Color'],
            'Strap Material': input_data['Strap_Material'],
            'Mechanism': input_data['Mechanism'],
            'Glass Type': input_data['Glass_Type'],
            'Dial Color': input_data['Dial_Color']
        }

        # 3. Create the single-row DataFrame
        # Use the global 'config' to ensure column order is 100% correct
        num_cols = config.columns.numerical_features
        cat_cols = config.columns.categorical_features

        df = pd.DataFrame(data_for_df, index=[0])
        df = df[num_cols + cat_cols]  # Enforce column order

        # 4. Call models from Memory (global 'models' dict)
        preprocessor = models["preprocessor"]
        model = models["model"]

        # 5. Run the Pipeline
        transformed_data = preprocessor.transform(df)
        prediction_log = model.predict(transformed_data)

        # 6. Inverse Log-Transform (and fix the NumPy bug)
        prediction_usd = np.expm1(prediction_log[0])
        prediction_usd_standard = float(prediction_usd)  # Convert from numpy.float32 to standard Python float

        return {"predicted_price_usd": round(prediction_usd_standard, 2)}

    except KeyError as e:
        logger.error(f"Prediction failed due to KeyError: {e}")
        return {"error": f"Missing a feature the model expects: {e}"}
    except Exception as e:
        logger.error(f"Prediction failed with an unexpected error: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    # This block is for direct execution (python app/main.py)
    # which is NOT the recommended way.
    logger.warning("Do not run this script directly with 'python app/main.py'.")
    logger.warning("To start the server, use the command: 'uvicorn app.main:app --reload'")
    uvicorn.run(app, host="127.0.0.1", port=8000)