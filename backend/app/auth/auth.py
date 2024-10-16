import os
import jwt
from dotenv import load_dotenv
from fastapi import HTTPException, status, Request
from datetime import datetime, timedelta
from typing import Optional
from schemas import Token

load_dotenv()

ALGORITHM = "HS256"
token_secret_key = os.getenv("TOKEN_SECRET_KEY")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token.

    Args:
        data (dict): The data to encode in the token.
        expires_delta (Optional[timedelta]): The time duration after which the token will expire.
                                            If not provided, the token will expire in 15 minutes.

    Returns:
        str: The encoded JWT token.
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, token_secret_key, algorithm=ALGORITHM)
    return encoded_jwt

def decode_access_token(token: str):
    """
    Decodes a JWT access token to extract the username and role.
    Args:
        token (str): The JWT access token to decode.
    Returns:
        dict: A dictionary containing the username and role.
    Raises:
        HTTPException: If the token has expired or is invalid.
    """
    try:
        payload = jwt.decode(token, token_secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"username": username, "role": role}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def generate_token_response(username: str, role: str):
    """
    Generates an access token for the given username and role.

    Args:
        username (str): The username for which the access token is generated.
        role (str): The role of the user (e.g., 'user', 'admin').

    Returns:
        Token: An object containing the access token and its type.
    """
    access_token_expires = timedelta(
        minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))
    )
    access_token = create_access_token(
        data={"sub": username, "role": role}, expires_delta=access_token_expires
    )

    return Token(access_token=access_token, token_type="bearer")

def get_authenticated_user(request: Request):
    """
    Get the authenticated user and their role from the access token.

    Args:
        request (Request): The incoming request containing the JWT token.

    Returns:
        dict: A dictionary containing 'username' and 'role'.
    """
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token = auth_header.split(" ")[1]
        payload = jwt.decode(token, token_secret_key, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None or role is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return {"username": username, "role": role}
    except HTTPException as e:
        raise HTTPException(
            status_code=e.status_code,
            detail=e.detail,
            headers=e.headers,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
