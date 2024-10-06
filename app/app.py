import logging
import os
import yaml
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
from db import connect_to_mongo, get_collection
from app.auth import generate_token_response, JWTBearer, get_authenticated_user
from utils import get_logger, hash_password, verify_password
from schemas import User, UserCredentials, ReviewInput
from utils import config


# Load configuration
with open(os.path.join('configs', 'config.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

load_dotenv()

version = "v1"

# Initialize FastAPI app
app = FastAPI(
    title="Text Sentiment Analysis API",
    description="An API to predict sentiment of text using an LSTM model.",
    version=version,
)

db = connect_to_mongo()
logger = get_logger("api")

# Load model and tokenizer
model_path = os.path.join(config["model"]["output_dir"], "bert_classifier.pth")
model = BertForSequenceClassification.from_pretrained(
    config["model"]["bert_model_name"], num_labels=config["model"]["num_classes"]
)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

tokenizer = BertTokenizer.from_pretrained(config["model"]["bert_model_name"])

# Logging setup
logging.basicConfig(
    filename="api_requests.log", level=logging.INFO, format="%(asctime)s - %(message)s"
)


@app.get("/")
def root():
    """
    The root endpoint.
    """
    return {"message": f"Hello, From Sentiment Analysis Application! {version}"}


@app.post("/register")
async def register(user: User):
    """
    Registers a new user.
    Args:
        user (UserCredentials): The user's registration details.

    Returns:
        dict: A dictionary containing the user's registration details.
    """

    collection = get_collection(db, "users")
    user.created_at = datetime.now()
    user.password = hash_password(user.password)
    result = collection.insert_one(dict(user))
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User registration failed",
        )

    return {"result": "User registered successfully"}


@app.post("/login")
async def login(credentials: UserCredentials):
    """
    Authenticates a user and generates an access token.
    Args:
        credentials (UserCredentials): The user's login credentials.

    Raises:
        HTTPException: If the username or password is incorrect.

    Returns:
        dict: A dictionary containing the access token.
    """
    collection = get_collection(db, "users")
    user = collection.find_one({"username": credentials.username})
    matched = verify_password(credentials.password, user["password"])
    if not user or not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return generate_token_response(user["username"])


@app.get("/admin", dependencies=[Depends(JWTBearer())])
async def admin_endpoint(request: Request):
    """
    Admin-only endpoint. Requires JWT token with admin role.
    """
    username = get_authenticated_user(request)
    collection = get_collection(db, "users")
    user = collection.find_one({"username": username})
    if user["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return {"message": f"Welcome, {username}! This is the admin area."}


@app.post("/predict", dependencies=[Depends(JWTBearer())])
async def predict(input_data: ReviewInput, request: Request):
    try:
        # Token verification
        username = get_authenticated_user(request)
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Token is valid, proceed with prediction
        inputs = tokenizer(
            input_data.review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=config["model"]["max_length"],
        )
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = "positive" if prediction == 1 else "negative"

        # Log the request
        logging.info(
            f"User: {username}, Review: {input_data.review}, Prediction: {sentiment}"
        )

        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
