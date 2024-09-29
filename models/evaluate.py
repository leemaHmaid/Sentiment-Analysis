from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def evaluate_lstm_model(model, X_test, y_test):
    """
    Evaluate a trained LSTM model and print metrics.

    Args:
        model (Sequential): Trained Keras model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        tuple: (accuracy, classification report)
    """
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report
