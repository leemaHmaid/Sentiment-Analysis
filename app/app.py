import jwt
import csv
import logging
import os
import yaml
import torch
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
from .security import authenticate_user
from datetime import datetime, timedelta


# Load configuration
with open(os.path.join('configs', 'config.yaml'), 'r') as config_file:
    config = yaml.safe_load(config_file)

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize FastAPI
app = FastAPI()

# Load model and tokenizer
model_path = os.path.join(config['model']['output_dir'], 'bert_classifier.pth')
model = BertForSequenceClassification.from_pretrained(config['model']['bert_model_name'], num_labels=config['model']['num_classes'])
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])

# Logging setup
logging.basicConfig(filename='api_requests.log', level=logging.INFO, format='%(asctime)s - %(message)s')

class ReviewInput(BaseModel):
    review: str

class TokenData(BaseModel):
    username: str

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

@app.post("/token")
async def login_for_access_token(username: str = Depends(authenticate_user)):
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

def log_request(username: str, review: str, prediction: str):
    with open('prediction_logs.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.utcnow(), username, review, prediction])

@app.post("/predict")
async def predict(input_data: ReviewInput, username: str = Depends(authenticate_user), token: str = Depends(login_for_access_token)):
    try:
        # Token verification
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        token_username: str = payload.get("sub")
        if token_username is None:
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
            max_length=config['model']['max_length']
        )
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = "positive" if prediction == 1 else "negative"

        # Log the request
        logging.info(f"User: {username}, Review: {input_data.review}, Prediction: {sentiment}")
        log_request(username, input_data.review, sentiment)

        return {"sentiment": sentiment}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))