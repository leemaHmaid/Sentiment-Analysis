from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_sentiment(model, tokenizer, text, max_len=200):
    """
    Predict the sentiment of a given text using the LSTM model.

    Args:
        model (Sequential): Trained Keras model.
        tokenizer (Tokenizer): Fitted tokenizer.
        text (str): Input text.
        max_len (int): Maximum sequence length.

    Returns:
        str: Predicted sentiment ('positive' or 'negative').
    """
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment
