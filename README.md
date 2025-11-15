# End-to-End Watch Price Regression (MLOps v5.0: Streamlit)

This is an end-to-end Machine Learning Operations (MLOps) pipeline built to "Google-level" standards, designed to predict the price of a watch based on its features.

This repository demonstrates a full MLOps lifecycle:
1.  **v1 (Baseline):** A robust 6-step pipeline was built to fix 5 "Cardinal Sins" (Data Leakage, etc.), achieving a baseline MAE of `$572`.
2.  **v2 (Optimization):** The baseline was optimized using **MLFlow** and **Optuna** to run 50+ experiments, finding a "champion" model that reduced the **MAE to $514.33**.
3.  **v3 (Serving):** The v2 "champion" model was packaged into a high-performance **FastAPI** server, built as an installable Python package.
4.  **v4 (Deployment):** The v3 FastAPI server was containerized using **Docker**, making the entire API portable and isolated.
5.  **v5 (Presentation):** This **Streamlit** dashboard was created as a "client-facing" UI that consumes the v4 Dockerized API.

---

## 🏛️ MLOps Architecture: Factory, Store & Showroom

This project is built on the core MLOps principle of **Separating Logic from Configuration**.

* **`config/config.yaml` (The Brain):**
    The "single source of truth." All file paths, outlier rules, imputation strategies, and final champion model parameters are defined here.
* **`src/` (The Factory):**
    Contains the 6-step pipeline (`src/pipeline/`) and the experiment lab (`src/run_tuning.py`). Its only job is to *produce* the final models.
* **`app/` (The Store):**
    Contains the **FastAPI** server (`app/main.py`). Its only job is to *serve* the models. This is the **Backend API**.
* **`dashboard/` (The Showroom):**
    Contains the **Streamlit** dashboard (`dashboard/app.py`). It is a "dumb" client that *consumes* the API. This is the **Frontend UI**.

---

## 🚀 Project Structure
```
watch-price-regression/

│

├── app/                      <- (v3: The "Store" - Backend API)

│   ├── init.py

│   ├── main.py               <- (FastAPI app definition)

│   └── schema.py             <- (Pydantic input schema)

│

├── config/                   <- (The "Brain": All config)

│   └── config.yaml

│

├── dashboard/                <- (v5: The "Showroom" - Frontend UI)

│   └── app.py                <- (Streamlit dashboard script)

│

├── data/                     <- (.gitignored: Raw & Processed Data)

├── models/                   <- (.gitignored: Trained v2 Models)

├── mlruns/                   <- (.gitignored: v2 MLFlow Logs)

├── notebooks/                <- (Initial R&D)

│

├── src/                      <- (v1-v2: The "Factory" - Training Code)

│   ├── init.py

│   ├── pipeline/

│   │   ├── ... (6 pipeline steps)

│   ├── run_tuning.py         <- (v2: Optuna + MLFlow experiment runner)

│   └── utils.py              <- (Helper function to load config)

│

├── .dockerignore             <- (v4: Tells Docker what not to copy)

├── .gitignore                <- (Tells Git what not to track)

├── Dockerfile                <- (v4: The API container "recipe")

├── environment.yml           <- (Conda environment dependencies)

├── setup.py                  <- (Makes 'src' and 'app' installable packages)

└── README.md                 <- (This file - The project user manual)
```

---

---

## 🛠️ Installation & Setup

Follow these steps to set up the project environment on your local machine.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/enesml/watch-price-regression.git
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
    This is the crucial step that makes your `src` and `app` code importable.
    ```bash
    pip install -e .
    ```

---

## ⚡ How to Use the Project (v5)

Running the v5 dashboard requires **two separate terminals** running at the same time: The "Store" (API) and the "Showroom" (UI).

### ➡️ Terminal 1: Run the API Server (v4 Docker)
This runs the v4 "Store" API inside its Docker container.
*(Note: We use port 8001 to avoid conflicts with common 'zombie' port 8000).*

```bash
# (Re-build the image if you changed the API code)
# docker build -t watch-api:v4 .

# Run the container (mapping host 8001 to container 8000)
docker run --rm -p 8001:8000 -v ${pwd}/models:/app/models watch-api:v4
```
This terminal will be busy running the API at http://127.0.0.1:8001.

### ➡️ Terminal 2: Run the Streamlit Dashboard (v5 UI)
```bash
# (Make sure you are in the 'watch-ml' conda environment)
conda activate watch-ml

# Run the Streamlit app
streamlit run dashboard/app.py
```
This will automatically open your browser to the Streamlit app (usually http://localhost:8501).

You can now interact with the UI, which will send live prediction requests to the API running in Terminal 1.

---

## (Optional) Development & Training
### Train the Champion Model (v1-v2)
This runs the full 6-step pipeline in sequence.

```bash
python src/pipeline/_01_data_ingestion.py
python src/pipeline/_02_outlier_removal.py
python src/pipeline/_03_imputation.py
python src/pipeline/_04_feature_creation.py
python src/pipeline/_05_data_transformation.py
python src/pipeline/_06_model_training.py
```

### Run MLFlow Experiments (v2)
To find better models:

```bash
# Terminal 1:
mlflow ui --port 5001

# Terminal 2:
python src/run_tuning.py
```

## 🧪 v6: Running the Automated Tests (QA)
This project is insured by Pytest to guarantee that the "Factory" (src/) and "Store" (app/) components are working correctly.

(Note: These tests require the models/ folder to be populated. Run the Training Pipeline (Step 1 below) at least once before testing.)

To run all automated tests (4 tests total):

```bash
# (Make sure you are in the 'watch-ml' conda environment)
pytest
```

### Expected Output (Success):

```bash
============================= test session starts ==============================
collected 4 items

tests\test_api.py ...                                                    [ 75%]
tests\test_pipeline.py .                                                 [100%]

============================== 4 passed in 1.20s ===============================
```

