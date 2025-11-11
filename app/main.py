import pandas as pd
import joblib
import numpy as np
import uvicorn
import logging
import sys
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import our 'schema' module from the 'app' folder
from app.schema import WatchFeatures

# --- Project Path Setup ---

try:
    from src.utils import load_config
except ImportError:
    logging.error("ERROR: src/utils.py not found. Please ensure that the home directory is added to sys.path.")
    sys.exit(1)

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading ---

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 'lifespan' event: Runs before the server starts.
    Loads models ONLY ONCE from dis-ten and assigns them to the 'models' dictionary.
    """
    global config
    try:
        # 1. Install "Brain" (make the config global so /predict can see it)
        config = load_config()

        # 2. Load the Preprocessor
        preprocessor_path = PROJECT_ROOT / config.paths.scaler
        models["preprocessor"] = joblib.load(preprocessor_path)

        # 3. Download "Champion Engine" (v2 Model)
        model_path = PROJECT_ROOT / config.paths.model
        models["model"] = joblib.load(model_path)

        logger.info(f"Models loaded successfully. Preprocessor: {preprocessor_path}, Model: {model_path}")

    except Exception as e:
        logger.error(f"Modeller yüklenirken FATAL HATA: {e}")
        raise

    yield

    # When the server shuts down...
    models.clear()
    logger.info("Models cleared from memory.")


# Initialize the FastAPI application with our 'lifespan' function
app = FastAPI(
    lifespan=lifespan,
    title="Wristwatch Price Prediction API (v2)",
    description="API that provides the MLOps v2 (Optuna-tuned XGBoost) model.",
    version="2.0.0"
)


@app.get("/", tags=["Health Check"])
def health_check():
    """Check if the server is up."""
    return {"status": "ok", "message": "API is running!"}


@app.post("/predict", tags=["Prediction"])
def predict(features: WatchFeatures):
    """
    Get time properties (JSON) and return price estimate (JSON).
    """

    try:
        # 1. Convert Pydantic schema to a dictionary (dict)
        input_data = features.model_dump()

        # 2. Translation: Translate Pydantic names to DataFrame column names
        # This creates exactly what the preprocessor (key) expects.
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

        # 3. Create a single-row DataFrame
        num_cols = config.columns.numerical_features
        cat_cols = config.columns.categorical_features

        df = pd.DataFrame(data_for_df, index=[0])
        df = df[num_cols + cat_cols]

        # 4. Recall Models from Memory (models dictionary)
        preprocessor = models["preprocessor"]
        model = models["model"]

        # 5. Run the Pipeline
        transformed_data = preprocessor.transform(df)
        prediction_log = model.predict(transformed_data)

        # 6. Undo Log-Transform (np.expm1)
        prediction_usd = np.expm1(prediction_log[0])

        prediction_usd_standard = float(prediction_usd)

        return {"predicted_price_usd": round(prediction_usd_standard, 2)}

    except KeyError as e:
        logger.error(f"KeyError during guess: {e}")
        return {"error": f"A feature expected by the model is missing: {e}"}
    except Exception as e:
        logger.error(f"ERROR during prediction: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)