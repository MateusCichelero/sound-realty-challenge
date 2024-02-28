import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

# Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Housing Prices ML Regression API"}


# Test health check endpoint
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# TODO: test api endpoints