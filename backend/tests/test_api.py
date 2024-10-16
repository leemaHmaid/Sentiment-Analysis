import pytest
from fastapi.testclient import TestClient
from app.app import app 
from db import connect_to_mongo, get_collection
from utils import hash_password
from datetime import datetime
from unittest.mock import MagicMock

# Initialize TestClient
client = TestClient(app)

# Connect to the test database
test_db = connect_to_mongo()

# Mock the model loading
@pytest.fixture(autouse=True)
def mock_model(mocker):
    mocker.patch('app.app.model.load_state_dict', return_value=None)
    mocker.patch('app.app.BertForSequenceClassification.from_pretrained', return_value=MagicMock())

# Test user data
@pytest.fixture(scope="module")
def test_users():
    users = [
        {
            "full_name": "Test User",
            "email": "testuser@example.com",
            "role": "user",
            "username": "testuser",
            "password": "testpassword"
        },
        {
            "full_name": "Admin User",
            "email": "adminuser@example.com",
            "role": "admin",
            "username": "adminuser",
            "password": "adminpassword"
        }
    ]
    # Register users
    for user in users:
        user_data = user.copy()
        user_data["password"] = hash_password(user_data["password"])
        get_collection(test_db, "users").insert_one({
            "username": user_data["username"],
            "full_name": user_data["full_name"],
            "email": user_data["email"],
            "role": user_data["role"],
            "password": user_data["password"],
            "created_at": datetime.utcnow()
        })
    yield users
    # Teardown: Remove test users
    usernames = [user["username"] for user in users]
    get_collection(test_db, "users").delete_many({"username": {"$in": usernames}})
    # Teardown: Remove sentiment histories
    get_collection(test_db, "sentiment_history").delete_many({"username": {"$in": usernames}})

@pytest.fixture(scope="module")
def get_tokens(test_users):
    tokens = {}
    for user in test_users:
        response = client.post("/login", json={
            "username": user["username"],
            "password": user["password"]
        })
        assert response.status_code == 200
        tokens[user["username"]] = response.json()["access_token"]
    return tokens

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, From Sentiment Analysis Application! v1"}

def test_register_endpoint(test_users):
    new_user = {
        "full_name": "Another User",
        "email": "anotheruser@example.com",
        "role": "user",
        "username": "anotheruser",
        "password": "anotherpassword"
    }
    response = client.post("/register", json=new_user)
    assert response.status_code == 200
    assert response.json() == {"result": "User registered successfully"}
    # Cleanup
    get_collection(test_db, "users").delete_one({"username": "anotheruser"})

def test_login_endpoint(get_tokens):
    # Already tested in get_tokens fixture
    assert "testuser" in get_tokens
    assert "adminuser" in get_tokens

def test_user_info_endpoint(get_tokens):
    token = get_tokens["testuser"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/user/info", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "testuser"
    assert data["role"] == "user"

def test_predict_endpoint_with_valid_data(get_tokens):
    token = get_tokens["testuser"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "review": "The movie was fantastic!"
    }
    response = client.post("/predict", headers=headers, json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["positive", "negative"]

def test_predict_endpoint_with_missing_data(get_tokens):
    token = get_tokens["testuser"]
    headers = {"Authorization": f"Bearer {token}"}
    payload = {}  # Missing "review" field
    response = client.post("/predict", headers=headers, json=payload)
    assert response.status_code == 422  # Unprocessable Entity

def test_predict_endpoint_with_invalid_token():
    invalid_headers = {
        "Authorization": "Bearer invalidtoken"
    }
    payload = {
        "review": "The movie was fantastic!"
    }
    response = client.post("/predict", headers=invalid_headers, json=payload)
    assert response.status_code in [401, 403]  # Depending on FastAPI response

def test_admin_history_endpoint_as_admin(get_tokens):
    token = get_tokens["adminuser"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin/history", headers=headers)
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_admin_history_endpoint_as_user(get_tokens):
    token = get_tokens["testuser"]
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/admin/history", headers=headers)
    assert response.status_code == 403  # Forbidden

def test_admin_history_endpoint_with_invalid_token():
    invalid_headers = {
        "Authorization": "Bearer invalidtoken"
    }
    response = client.get("/admin/history", headers=invalid_headers)
    assert response.status_code == 403  # Forbidden
