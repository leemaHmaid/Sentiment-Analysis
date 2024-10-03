import os
import pandas as pd
from dotenv import load_dotenv
from keras.preprocessing.sequence import pad_sequences
from utils.logger import get_logger 
from pydantic import BaseModel
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import (
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from passlib.context import CryptContext
from typing import Union
from datetime import datetime, timedelta, timezone
import jwt
from jwt import PyJWTError
from tensorflow.keras.models import load_model
import joblib

load_dotenv()
# JWT settings
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI app
app = FastAPI()
logger = get_logger('api')

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Load model and tokenizer
model = load_model('models/lstm_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

# In-memory user database (for demonstration purposes)
users_db = {
    "user1": {
        "username": "user1",
        "full_name": "User One",
        "password": pwd_context.hash("user1password"),
        "role": "user",
    },
    "admin": {
        "username": "admin",
        "full_name": "Admin User",
        "password": pwd_context.hash("adminpassword"),
        "role": "admin",
    },
}

# Pydantic models
class Review(BaseModel):
    text: str

# Helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def authenticate_user(username: str, password: str):
    user = users_db.get(username)
    if not user or not verify_password(password, user["password"]):
        return None
    return user

def create_access_token(data: dict, expires_delta: Union[timedelta, None] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
            )
        return {"username": username, "role": role}
    except PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )

# API endpoints
@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate user and provide JWT token.
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"], "role": user["role"]},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict")
async def predict_sentiment(
    review: Review, token: str = Depends(oauth2_scheme)
):
    """
    Predict sentiment of the provided text. Requires JWT token.
    """
    token_data = decode_token(token)
    sentiment = predict_sentiment_lstm(model, tokenizer, review.text)
    # Log the input, prediction, and username
    log_prediction(review.text, sentiment, username=token_data["username"])
    return {"sentiment": sentiment, "user": token_data["username"]}

@app.get("/admin")
async def admin_endpoint(token: str = Depends(oauth2_scheme)):
    """
    Admin-only endpoint. Requires JWT token with admin role.
    """
    token_data = decode_token(token)
    if token_data["role"] != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Not enough permissions"
        )
    return {"message": f"Welcome, {token_data['username']}! This is the admin area."}

# Utility functions
def predict_sentiment_lstm(model, tokenizer, text, max_len=200):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment

def log_prediction(input_text, prediction, username, log_file='logs/predictions.csv'):
    """
    Log the input text, prediction, and username to a CSV file.
    """
    data = {
        'timestamp': [pd.Timestamp.now()],
        'username': [username],
        'input_text': [input_text],
        'prediction': [prediction],
    }
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.isfile(log_file):
        df.to_csv(log_file, index=False)
    else:
        df.to_csv(log_file, mode='a', header=False, index=False)





