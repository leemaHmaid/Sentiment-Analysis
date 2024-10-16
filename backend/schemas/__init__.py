from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserCredentials(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class ReviewInput(BaseModel):
    review: str

class User(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    role: str
    created_at: datetime

class UserCreate(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    role: str
    password: str

class PredictResponse(BaseModel):
    sentiment: str
    confidence: float

class ReviewHistory(BaseModel):
    username: str
    review: str
    sentiment: str
    confidence: float
    timestamp: datetime

class RequestLog(BaseModel):
    username: str
    user_id: str
    time: datetime
    endpoint: str
    status_code: int
    request_body: dict
    request_headers: dict
