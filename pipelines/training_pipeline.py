from src.features.feature_extraction import tokenize_and_pad_sequences
from src.models.train import train_lstm_model
from src.models.evaluate import evaluate_lstm_model
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import os

def run(config):
    """
    Run the training pipeline for the LSTM model.

    Args:
        config (Config): Configuration object.
    """
    # Load processed data
    data_path = config.get('data', 'processed_data_path')
    df = pd.read_csv(data_path)
    texts = df['clean_review'].tolist()
    labels = df['sentiment'].values

    # Tokenize and pad sequences
    X, tokenizer = tokenize_and_pad_sequences(texts, max_len=200, num_words=5000)
    y = labels

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get vocabulary size
    vocabulary_size = len(tokenizer.word_index)

    # Train LSTM model
    model = train_lstm_model(X_train, y_train, vocabulary_size, max_len=200)

    # Evaluate model
    accuracy, report = evaluate_lstm_model(model, X_test, y_test)
    print(f"Accuracy: {accuracy}")
    print(report)

    # Save model and tokenizer
    model_path = config.get('model', 'model_path')
    tokenizer_path = config.get('model', 'tokenizer_path')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    joblib.dump(tokenizer, tokenizer_path)
