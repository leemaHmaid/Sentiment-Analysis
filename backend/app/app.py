import logging
import os
import yaml
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status, Request, APIRouter
from transformers import BertTokenizer, BertForSequenceClassification
from datetime import datetime
from db import connect_to_mongo, get_collection
from app.auth import generate_token_response, get_authenticated_user
from app.auth import JWTBearer, AdminJWTBearer
from utils import get_logger, hash_password, verify_password
from schemas import (
    UserCreate,
    UserCredentials,
    Token,
    ReviewInput,
    PredictResponse,
    ReviewHistory,
    User
)
from pydantic import BaseModel, EmailStr
from typing import List

# Load configuration
with open(os.path.join('backend/configs', 'config.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

load_dotenv()

version = "v1"

# Initialize FastAPI app
app = FastAPI(
    title="Text Sentiment Analysis API",
    description="An API to predict sentiment of text using a BERT model.",
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

# CORS Configuration (if needed)
from fastapi.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:8501",  # Streamlit frontend
    # Add other allowed origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specified origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Router
router = APIRouter()

@app.get("/")
def root():
    """
    The root endpoint.
    """
    return {"message": f"Hello, From Sentiment Analysis Application! {version}"}


@router.post("/register", response_model=dict)
async def register(user: UserCreate):
    """
    Registers a new user.
    Args:
        user (UserCreate): The user's registration details.

    Returns:
        dict: A dictionary containing the user's registration result.
    """

    collection = get_collection(db, "users")
    
    # Check if username or email already exists
    existing_user = collection.find_one({"username": user.username})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists",
        )
    
    existing_email = collection.find_one({"email": user.email})
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )
    
    # Set created_at server-side
    user_data = user.dict()
    user_data["created_at"] = datetime.utcnow()
    user_data["password"] = hash_password(user.password)
    
    result = collection.insert_one(user_data)
    if not result.inserted_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User registration failed",
        )

    logger.info(f"New user registered: {user.username}")
    return {"result": "User registered successfully"}


@router.post("/login", response_model=Token)
async def login(credentials: UserCredentials):
    """
    Authenticates a user and generates an access token.
    Args:
        credentials (UserCredentials): The user's login credentials.

    Raises:
        HTTPException: If the username or password is incorrect.

    Returns:
        Token: A Pydantic model containing the access token and its type.
    """
    collection = get_collection(db, "users")
    user = collection.find_one({"username": credentials.username})
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    matched = verify_password(credentials.password, user["password"])
    if not matched:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = generate_token_response(user["username"], user["role"])
    return token


@router.get("/user/info", response_model=User, dependencies=[Depends(JWTBearer())])
async def get_user_info_endpoint(request: Request):
    """
    Retrieves the authenticated user's information.
    """
    user_info = get_authenticated_user(request)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    collection = get_collection(db, "users")
    user = collection.find_one({"username": user_info["username"]})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )
    
    # Exclude sensitive information like password
    user_data = {
        "username": user["username"],
        "full_name": user["full_name"],
        "email": user["email"],
        "role": user["role"],
        "created_at": user["created_at"]
    }
    
    return user_data


@router.get("/admin/history", response_model=List[ReviewHistory], dependencies=[Depends(AdminJWTBearer())])
async def get_all_sentiment_histories():
    """
    Admin endpoint to retrieve all sentiment analysis histories of all users.
    """
    collection = get_collection(db, "sentiment_history")
    histories = list(collection.find({}, {"_id": 0}))
    return histories


@router.post("/predict", response_model=PredictResponse, dependencies=[Depends(JWTBearer())])
async def predict(input_data: ReviewInput, request: Request):
    """
    Predicts the sentiment of the provided review.
    Args:
        input_data (ReviewInput): The review text to analyze.
        request (Request): The incoming request for authentication.

    Returns:
        PredictResponse: A Pydantic model containing the sentiment and confidence score.
    """
    try:
        # Token verification
        user_info = get_authenticated_user(request)
        username = user_info["username"]
        role = user_info["role"]
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
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = torch.max(probabilities, dim=1)
        confidence_percentage = round(confidence.item() * 100, 2)
        sentiment = "positive" if prediction.item() == 1 else "negative"

        # Log the request
        logging.info(
            f"User: {username}, Review: {input_data.review}, Prediction: {sentiment}, Confidence: {confidence_percentage}%"
        )

        # Save to sentiment_history collection
        sentiment_collection = get_collection(db, "sentiment_history")
        sentiment_data = {
            "username": username,
            "review": input_data.review,
            "sentiment": sentiment,
            "confidence": confidence_percentage,
            "timestamp": datetime.utcnow()
        }
        sentiment_collection.insert_one(sentiment_data)

        return PredictResponse(sentiment=sentiment, confidence=confidence_percentage)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

app.include_router(router)