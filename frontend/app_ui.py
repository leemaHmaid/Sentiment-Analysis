import streamlit as st
import requests
import json
from typing import Optional
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd

# Load environment variables
load_dotenv()

# Configuration
API_URL = os.getenv("API_BASE_URL")

# Utility Functions
def register_user(full_name: str, email: str, role: str, username: str, password: str) -> bool:
    url = f"{API_URL}/register"
    payload = {
        "full_name": full_name,
        "email": email,
        "role": role,
        "username": username,
        "password": password
    }
    try:
        with st.spinner("Registering..."):
            response = requests.post(url, json=payload)
        if response.status_code == 200:
            return True
        else:
            st.error(f"Registration failed: {response.json().get('detail', 'Unknown error')}")
            return False
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Please ensure FastAPI is running.")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return False

def login_user(username: str, password: str) -> Optional[dict]:
    url = f"{API_URL}/login"
    payload = {"username": username, "password": password}
    try:
        with st.spinner("Logging in..."):
            response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            token_type = data.get("token_type")
            return {"access_token": token, "token_type": token_type}
        else:
            st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Please ensure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_user_info(token: str) -> Optional[dict]:
    url = f"{API_URL}/user/info"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with st.spinner("Fetching user info..."):
            response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to retrieve user information.")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Please ensure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def predict_sentiment(token: str, review: str) -> Optional[dict]:
    url = f"{API_URL}/predict"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"review": review}
    try:
        with st.spinner("Analyzing sentiment..."):
            response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            sentiment = response.json().get("sentiment")
            confidence = response.json().get("confidence")  # Now returned by the backend
            return {"sentiment": sentiment, "confidence": confidence}
        else:
            st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Please ensure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def get_all_sentiment_histories(token: str) -> Optional[list]:
    url = f"{API_URL}/admin/history"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with st.spinner("Fetching all sentiment histories..."):
            response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            st.error("Failed to retrieve sentiment histories.")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Failed to connect to the backend server. Please ensure FastAPI is running.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

def assess_password_strength(password: str) -> str:
    """Simple password strength checker."""
    if len(password) < 6:
        return "Weak"
    elif len(password) < 10:
        return "Moderate"
    else:
        return "Strong"

# Initialize Session State
if 'token' not in st.session_state:
    st.session_state.token = None

if 'username' not in st.session_state:
    st.session_state.username = ""

if 'role' not in st.session_state:
    st.session_state.role = ""

if 'sentiment_history' not in st.session_state:
    st.session_state.sentiment_history = []

if 'all_sentiment_histories' not in st.session_state:
    st.session_state.all_sentiment_histories = []

if 'theme' not in st.session_state:
    st.session_state.theme = 'Light'

# Streamlit Interface
def main():
    # Configure page layout
    st.set_page_config(
        page_title="📝 Sentiment Analysis Application",
        # layout="wide",
        # initial_sidebar_state="expanded",
    )

    # Title and Description
    st.title("📝 Sentiment Analysis Application") 
    st.markdown("""
    Welcome to the **Text Sentiment Analysis App**! This application allows you to analyze the sentiment of your text reviews.
    Whether you're gauging customer feedback or analyzing social media posts, our tool provides quick and accurate sentiment insights.
    """)

    # Sidebar Instructions, How to Use, and Theme Toggle
    menu = ["Home", "Login", "Register", "Predict Sentiment"]
    if st.session_state.role == "admin":
        menu.append("Admin")
    # choice = st.sidebar.radio("Menu", menu)
    choice = st.sidebar.selectbox("Menu", menu)

    # How to Use Section
    with st.sidebar.expander("ℹ️ How to Use"):
        st.markdown("""
        **Steps to use the app:**
        1. **Register**: Create a new account by providing a username and password.
        2. **Login**: Access your account using your credentials.
        3. **Predict Sentiment**: Enter the text you want to analyze and get the sentiment result.
        4. **Logout**: Securely log out of your account when done.
        """)

    # Theme Toggle
    st.sidebar.markdown("---")
    theme = st.sidebar.selectbox("Theme", ["Light", "Dark"], index=0 if st.session_state.theme == 'Light' else 1)
    if st.session_state.theme != theme:
        st.session_state.theme = theme
        if theme == "Dark":
            st.markdown(
                """
                <style>
                body {
                    background-color: #2e2e2e;
                    color: white;
                }
                </style>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <style>
                body {
                    background-color: white;
                    color: black;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

    # Menu Sections
    if choice == "Home":
        if st.session_state.token:
            st.success(f"Logged in as **{st.session_state.username}**")
            st.markdown("Navigate through the menu to analyze text or manage your account.")
        else:
            st.info("Please **login** or **register** to use the app.")

    elif choice == "Register":
        st.subheader("📝 Create a New Account")
        with st.form("registration_form"):
            full_name = st.text_input("Full Name", max_chars=100)
            email = st.text_input("Email", max_chars=100)
            role = st.selectbox("Role", ["user", "admin"])
            username = st.text_input("Username", max_chars=50)
            password = st.text_input("Password", type='password')
            confirm_password = st.text_input("Confirm Password", type='password')
            password_strength = assess_password_strength(password)
            st.write(f"**Password Strength:** {password_strength}")
            submitted = st.form_submit_button("Register")

            if submitted:
                if password != confirm_password:
                    st.error("❌ Passwords do not match.")
                elif not full_name or not email or not role or not username or not password:
                    st.error("❌ Please provide all required fields.")
                else:
                    success = register_user(full_name, email, role, username, password)
                    if success:
                        st.success("✅ Registration successful! You can now log in.")

    elif choice == "Login":
        st.subheader("🔑 Login to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type='password')
            submitted = st.form_submit_button("Login")

            if submitted:
                login_response = login_user(username, password)
                if login_response:
                    st.session_state.token = login_response["access_token"]
                    st.session_state.username = username
                    user_info = get_user_info(st.session_state.token)
                    if user_info:
                        st.session_state.role = user_info.get("role", "")
                        st.success(f"✅ Logged in as **{st.session_state.username}**")
                    else:
                        st.error("Failed to retrieve user information.")

    elif choice == "Predict Sentiment":
        if st.session_state.token:
            # st.subheader("📊 Enter Text for Sentiment Analysis")
            with st.form("sentiment_form"):
                review = st.text_area("Your Review", height=200)
                submitted = st.form_submit_button("Predict")

                if submitted:
                    if not review.strip():
                        st.error("❌ Please enter some text to analyze.")
                    else:
                        prediction = predict_sentiment(st.session_state.token, review)
                        if prediction:
                            sentiment = prediction.get("sentiment")
                            confidence = prediction.get("confidence")
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            st.session_state.sentiment_history.append({
                                "review": review,
                                "sentiment": sentiment,
                                "confidence": confidence,
                                "timestamp": timestamp
                            })
                            if sentiment.lower() == "positive":
                                st.success(f"😊 Positive (Confidence: {confidence}%)" if confidence is not None else "😊 Positive")
                            elif sentiment.lower() == "negative":
                                st.error(f"😞 Negative (Confidence: {confidence}%)" if confidence is not None else "😞 Negative")
                            else:
                                st.info(f"🔍 Sentiment: {sentiment} (Confidence: {confidence}%)" if confidence is not None else f"🔍 Sentiment: {sentiment}")

            # Display Sentiment History
            if st.session_state.sentiment_history:
                st.markdown("---")
                st.subheader("📚 Your Sentiment Analysis History")
                history = st.session_state.sentiment_history.copy()
                history.reverse()  # Show latest first
                for entry in history:
                    with st.expander(f"{entry['timestamp']}"):
                        st.write(f"**Review:** {entry['review']}")
                        sentiment = entry["sentiment"].capitalize()
                        confidence = entry["confidence"]
                        if sentiment.lower() == "positive":
                            sentiment_color = "green"
                            emoji = "😊"
                        elif sentiment.lower() == "negative":
                            sentiment_color = "red"
                            emoji = "😞"
                        else:
                            sentiment_color = "blue"
                            emoji = "🔍"
                        if confidence is not None:
                            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{emoji} {sentiment}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Confidence:** {confidence}%")
                            st.progress(confidence / 100)
                        else:
                            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{emoji} {sentiment}</span>", unsafe_allow_html=True)
                            st.write("**Confidence:** Not Available")
        else:
            st.warning("⚠️ You need to **log in** to perform sentiment analysis.")

    elif choice == "Admin":
        if st.session_state.role == "admin":
            st.subheader("🔍 All Sentiment Analysis Histories")
            with st.spinner("Loading..."):
                all_histories = get_all_sentiment_histories(st.session_state.token)
            if all_histories:
                df = pd.DataFrame(all_histories)
                st.dataframe(df)
            else:
                st.info("No sentiment analysis histories available.")
        else:
            st.error("You do not have permission to access this section.")

    # Logout Button and Profile Section
    if st.session_state.token:
        st.sidebar.markdown("---")
        with st.sidebar.expander("👤 Profile"):
            st.write(f"**Username:** {st.session_state.username}")
            st.write(f"**Role:** {st.session_state.role}")
            if st.session_state.sentiment_history:
                st.write(f"**Total Analyses:** {len(st.session_state.sentiment_history)}")
            else:
                st.write("**Total Analyses:** 0")
        if st.sidebar.button("🚪 Logout"):
            st.session_state.token = None
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.sentiment_history = []
            st.session_state.all_sentiment_histories = []
            st.success("✅ Logged out successfully.")

    # Footer
    st.markdown("""
        <style>
            footer {visibility: hidden;}
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: white;
                color: black;
                text-align: center;
            }
        </style>
        <div class="footer">
            <br />
            <div align="center">
                <a href="https://pmensah28.github.io/">
                    <img src="https://img.shields.io/badge/-Website-4B9AE5?style=flat&logo=Website&logoColor=white" alt="Website">
                </a>
                <a href="https://www.linkedin.com/in/prince-mensah/">
                    <img src="https://img.shields.io/badge/-LinkedIn-306EA8?style=flat&logo=Linkedin&logoColor=white" alt="LinkedIn">
                </a>
                <a href="https://github.com/pmensah28">
                    <img src="https://img.shields.io/badge/-GitHub-2F2F2F?style=flat&logo=github&logoColor=white" alt="GitHub">
                </a>
                <a href="https://www.kaggle.com/pmensah1">
                    <img src="https://img.shields.io/badge/-Kaggle-5DB0DB?style=flat&logo=Kaggle&logoColor=white" alt="Kaggle">
                </a>
                <a href="mailto:pmensah@aimsammi.org">
                    <img src="https://img.shields.io/badge/-Email-676767?style=flat&logo=google-scholar&logoColor=white" alt="Email Me">
                </a>
                <a href="https://www.buymeacoffee.com/pmensah">
                    <img src="https://img.shields.io/badge/-Buy_me_a_tea-yellow?style=flat&logo=buymeacoffee&logoColor=white" alt="Buy me a tea">
                </a>
                <a href="https://github.com/pmensah28/github-profile-views-counter">
                    <img src="https://komarev.com/ghpvc/?username=pmensah28" alt="GitHub Profile Views">
                </a>
                <a href="https://github.com/pmensah28?tab=followers">
                    <img src="https://img.shields.io/github/followers/pmensah28?label=Followers&style=social" alt="GitHub Badge">
                </a>
            </div>
                <br />
             <p>Copyright © Prince Mensah 2024 - Powered by Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
