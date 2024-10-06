import os
import joblib
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from utils.logger import get_logger
from tensorflow.keras.models import load_model
from db import connect_to_mongo, get_collection
from schemas import User, UserCredentials, Review
from keras.preprocessing.sequence import pad_sequences
from utils.hash_password import hash_password, verify_password
from fastapi import FastAPI, Depends, HTTPException, status, Request
from app.auth import generate_token_response, get_authenticated_user, JWTBearer


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
model = load_model("models/lstm_model.h5")
tokenizer = joblib.load("models/tokenizer.pkl")

# Pydantic models


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
            detail="Incorrect username or password",
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
async def predict_sentiment(review: Review, request: Request):
    """
    Predict sentiment of the provided text. Requires JWT token.
    """
    username = get_authenticated_user(request)
    sentiment = predict_sentiment_lstm(model, tokenizer, review.text)
    # Log the input, prediction, and username
    log_prediction(review.text, sentiment, username=username)
    return {"sentiment": sentiment, "user": username}


# Utility functions
def predict_sentiment_lstm(model, tokenizer, text, max_len=200):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment


def log_prediction(input_text, prediction, username, log_file="logs/predictions.csv"):
    """
    Log the input text, prediction, and username to a CSV file.
    """
    data = {
        "timestamp": [pd.Timestamp.now()],
        "username": [username],
        "input_text": [input_text],
        "prediction": [prediction],
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.isfile(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode="a", header=False, index=False)
