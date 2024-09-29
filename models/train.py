import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_lstm_model(X_train, y_train, vocabulary_size, max_len=200):
    """
    Train an LSTM model.

    Args:
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        vocabulary_size (int): Size of the vocabulary.
        max_len (int): Maximum sequence length.

    Returns:
        Sequential: Trained LSTM model.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocabulary_size + 1, output_dim=128, input_length=max_len))
    model.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=512, validation_split=0.1)
    return model
