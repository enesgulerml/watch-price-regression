# End-to-End Watch Price Regression (MLOps v3.0: API)

This is an end-to-end Machine Learning Operations (MLOps) pipeline built to "Google-level" standards, designed to predict the price of a watch based on its features.

This repository demonstrates a full MLOps lifecycle:
1.  **v1 (Baseline):** A robust 6-step pipeline was built to fix a flawed prototype (fixing 5 "Cardinal Sins" like Data Leakage), achieving a baseline MAE of `$572`.
2.  **v2 (Optimization):** The baseline was optimized using **MLFlow** and **Optuna** to run 50+ experiments, finding a "champion" model that reduced the **MAE to $514.33**.
3.  **v3 (Serving):** The v2 "champion" model and preprocessor were packaged into a high-performance **FastAPI** server, ready for deployment.

---

## 🏛️ MLOps Architecture: The "Factory" & The "Store"

This project is built on the core MLOps principle of **Separating Logic from Configuration**.

* **`config/config.yaml` (The Brain):**
    The "single source of truth." All file paths, outlier rules, imputation strategies, and final champion model parameters (found by Optuna) are defined here.
* **`src/` (The Factory):**
    Contains the 6-step pipeline (`src/pipeline/`) and the experiment lab (`src/run_tuning.py`). Its only job is to *produce* the final models (`model.joblib`, `scaler.joblib`).
* **`app/` (The Store):**
    Contains the **FastAPI** server (`app/main.py`) and data schema (`app/schema.py`). Its only job is to *use* (serve) the models produced by the "Factory".

---

## 🚀 Project Structure
```
watch-price-regression/

│

├── app/ <- (v3: The "Store" - FastAPI Server)

│ ├── init.py

│ ├── main.py <- (FastAPI app definition, /predict endpoint)

│ └── schema.py <- (Pydantic input validation schema)

│

├── config/

│ └── config.yaml <- (The "Brain": All paths, rules, params)

│

├── data/

│ ├── raw/ <- (Raw data, .gitignored)

│ ├── processed/ <- (Intermediate pipeline steps, .gitignored)

│ └── final/ <- (Train/Test sets, .gitignored)

│

├── models/

│ ├── scaler.joblib <- (The fitted preprocessor, .gitignored)

│ └── model.joblib <- (The v2 "champion" model, .gitignored)

│

├── mlruns/ <- (v2: MLFlow experiment logs, .gitignored)

│

├── notebooks/

│ └── 01-eda.ipynb <- (Initial R&D and scratchpad)

│

├── src/ <- (v1-v2: The "Factory" - Training Code)

│ ├── init.py

│ ├── pipeline/

│ │ ├── _01_data_ingestion.py

│ │ ├── _02_outlier_removal.py

│ │ ├── _03_imputation.py

│ │ ├── _04_feature_creation.py

│ │ ├── _05_data_transformation.py

│ │ └── _06_model_training.py

│ ├── run_tuning.py <- (v2: Optuna + MLFlow experiment runner)

│ └── utils.py <- (Helper function to load config)

│

├── .gitignore <- (.gitignored mlruns, data, models etc.)

├── environment.yml <- (Conda environment dependencies)

├── setup.py <- (Makes 'src' and 'app' installable packages)

└── README.md <- (This file - The project user manual)
```

---
## 🛠️ Installation & Setup

Follow these steps to set up the project environment on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/enesml/watch-price-regression.git](https://github.com/enesml/watch-price-regression.git)
    cd watch-price-regression
    ```

2.  **Download the Data:**
    * Go to the [Kaggle Dataset Page](https://www.kaggle.com/datasets/enesml/artificial-watch-price-prediction-dataset).
    * Download the `.csv` file.
    * Place it in `data/raw/` and rename it to `watch_price_raw.csv` (or update `config.yaml`).

3.  **Create Conda Environment:**
    ```bash
    conda env create -f environment.yml
    conda activate watch-ml
    ```

4.  **Install the Project Package:**
    This is the crucial step that makes your `src` and `app` code importable without hacks.
    ```bash
    pip install -e .
    ```

---

## ⚡ How to Use the Project

### 1. (v1-v2) Train the Champion Model
This runs the full 6-step pipeline using the "champion" parameters in `config.yaml` and saves the final models to the `models/` directory.

```bash
# Run the final step (which uses the outputs of the previous 5)
python src/pipeline/_06_model_training.py
```

* v1 Baseline (MAE): $572.25
* v2 Champion (MAE): $514.33

### 2. (Optional) Run Your Own v2 Experiments
To run your own 50-trial optimization (and see MLFlow in action):

```bash
# In Terminal 1 (The Lab Notebook):
mlflow ui --port 5001

# In Terminal 2 (The Scientist):
python src/run_tuning.py
```
Go to http://127.0.0.1:5001 to see the 50 experiments logged live.

### 3. (v3) Run the API Server (Local)
This loads the v2 champion models (model.joblib, scaler.joblib) and serves them via a FastAPI endpoint.

```bash
uvicorn app.main:app --reload --port 8000
```
This will start the server on http://127.0.0.1:8000.

### 4. Test the API
Go to your browser and open the automatic Swagger documentation:

* http://localhost:8000/docs

You can use the "Try it out" button on the /predict endpoint to send a test JSON and get a live price prediction.

## (v4) Build & Run the API Server (Docker)
This is the "production" way. It builds the v3 API into a self-contained container.

**Step 3.1:** Build the Docker Image This command reads the Dockerfile and builds your portable API image.

```bash
docker build -t watch-api:v4 .
```

**Step 3.2:** Run the Docker Container This command runs the image and "mounts" (connects) your local models/ folder to the container's /app/models/ folder.

```bash
docker run --rm -p 8000:8000 -v ${pwd}/models:/app/models watch-api:v4
```

**Step 3.3:** Test the Dockerized API Go to your browser and open the automatic Swagger documentation served from the container:

* http://127.0.0.1:8000/docs

You can now use the "Try it out" button to get live predictions from your Dockerized v4 API.