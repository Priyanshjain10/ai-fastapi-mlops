import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct structure"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "status" in data
    assert "version" in data
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "version" in data


def test_nlp_prediction_valid():
    """Test NLP prediction with valid input"""
    response = client.post(
        "/predict/nlp",
        json={"text": "This is a test", "task": "sentiment"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "model" in data
    assert "inference_time_ms" in data
    assert "request_id" in data


def test_nlp_prediction_empty_text():
    """Test NLP prediction with empty text"""
    response = client.post(
        "/predict/nlp",
        json={"text": "   ", "task": "sentiment"}
    )
    assert response.status_code == 422  # Validation error


def test_nlp_prediction_too_long():
    """Test NLP prediction with text exceeding max length"""
    long_text = "a" * 6000  # Exceeds 5000 char limit
    response = client.post(
        "/predict/nlp",
        json={"text": long_text, "task": "sentiment"}
    )
    assert response.status_code == 422  # Validation error
