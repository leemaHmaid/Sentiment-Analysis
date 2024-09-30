from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import joblib

class Review(BaseModel):
    text: str

app = FastAPI()

# Load model and tokenizer
model = load_model('models/lstm_model.h5')
tokenizer = joblib.load('models/tokenizer.pkl')

@app.post("/predict")
def predict_sentiment(review: Review):
    try:
        sentiment = predict_sentiment_lstm(model, tokenizer, review.text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_sentiment_lstm(model, tokenizer, text, max_len=200):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment