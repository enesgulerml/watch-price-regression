import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# --- Project Path Setup ---
# This is crucial for pytest to find the 'app' and 'src' packages
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import the FastAPI app object from our 'Store'
from app.main import app


# --- Test Fixtures (Reusable Test Client) ---

@pytest.fixture(scope="module")
def client():
    """
    Create a TestClient for our FastAPI app.
    This client "mocks" the server without needing 'uvicorn' or Docker.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_payload():
    """
    A valid JSON payload that matches app/schema.py.
    """
    return {
        "Case_Diameter": 40.5,
        "Water_Resistance": 10,
        "Warranty_Years": 2,
        "Weight_g": 85.0,
        "Brand": "Seiko",
        "Gender": "Male",
        "Case_Color": "Silver",
        "Glass_Shape": "Flat",
        "Origin": "Japan",
        "Case_Material": "Steel",
        "Additional_Feature": "Luminous",
        "Strap_Color": "Black",
        "Strap_Material": "Leather",
        "Mechanism": "Automatic",
        "Glass_Type": "Sapphire",
        "Dial_Color": "Blue"
    }


# --- Tests ---

def test_health_check(client):
    """
    Test the root endpoint (/) for a '200 OK' status.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running!"}


def test_prediction_success(client, valid_payload):
    """
    Test the /predict endpoint with valid data.
    We expect a '200 OK' and a valid price.
    """
    response = client.post("/predict", json=valid_payload)

    # 1. Check Status Code
    assert response.status_code == 200

    # 2. Check Response Body
    data = response.json()
    assert "predicted_price_usd" in data

    # 3. Check Value (Sanity Check)
    price = data["predicted_price_usd"]
    assert isinstance(price, float)
    assert 100 < price < 100000  # Same sanity check as test_pipeline


def test_prediction_invalid_data(client):
    """
    Test the /predict endpoint with invalid data (missing 'Brand').
    We expect a '422 Unprocessable Entity' error from Pydantic.
    """
    invalid_payload = {
        "Case_Diameter": 40.5,
        "Water_Resistance": 10
        # ... all other fields are missing
    }

    response = client.post("/predict", json=invalid_payload)

    # 1. Check Status Code
    assert response.status_code == 422  # 422 = Unprocessable Entity

    # 2. Check Response Body
    data = response.json()
    assert "detail" in data
    assert data["detail"][0]["msg"] == "Field required"
    assert data["detail"][0]["loc"] == ["body", "Warranty_Years"]  # Example check