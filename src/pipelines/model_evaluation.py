import os
import yaml
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from src.utils.logger import setup_logger
from src.utils.config import load_config
import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm

def run_model_evaluation():
    # Load configuration
    config = load_config(os.path.join('backend/configs', 'config.yaml'))
    
    # Set up logging
    logger = setup_logger(config['logging']['log_file'], config['logging']['level'])
    
    logger.info("\n\nStarting Model Evaluation...")
    
    try:
        # Set MLflow tracking URI and experiment
        mlflow.set_tracking_uri(config['model']['mlflow_tracking_uri'])
        mlflow.set_experiment(config['model']['mlflow_experiment_name'])
        
        # Load the trained model
        model_path = os.path.join(config['model']['output_dir'], 'bert_classifier.pth')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}")
        
        logger.info(f"Loading trained model from {model_path}")
        model = BertForSequenceClassification.from_pretrained(config['model']['bert_model_name'], num_labels=config['model']['num_classes'])
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Load the test dataset
        test_data_path = os.path.join(config['data']['processed_data_path'], 'test.csv')
        if not os.path.exists(test_data_path):
            raise FileNotFoundError(f"Test dataset not found at {test_data_path}")
        
        logger.info(f"\nLoading test data from {test_data_path}")
        test_df = pd.read_csv(test_data_path)
        test_df['sentiment'] = test_df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Initialize tokenizer
        tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
        
        # Tokenize the test data
        logger.info("Tokenizing test data.")
        test_encodings = tokenizer(
            test_df['review'].tolist(),
            truncation=True,
            padding=True,
            max_length=config['model']['max_length']
        )
        
        # Create a simple dataset
        class SentimentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

            def __len__(self):
                return len(self.labels)
        
        test_dataset = SentimentDataset(test_encodings, test_df['sentiment'].tolist())
        
        # Create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
        # Perform evaluation
        all_preds = []
        all_labels = []
        
        logger.info("Performing model evaluation.")
        progress_bar = tqdm(test_loader, desc="Evaluating")
        with torch.no_grad():
            for batch in progress_bar:
                inputs = {k: v for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].numpy()
                outputs = model(**inputs)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)
                progress_bar.set_postfix({'batch_accuracy': accuracy_score(labels, preds)})
        
        # Calculate metrics
        logger.info("Calculating evaluation metrics.")
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovo', average='weighted') if config['model']['num_classes'] > 2 else roc_auc_score(all_labels, all_preds)
        report = classification_report(all_labels, all_preds)
        
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"ROC-AUC Score: {roc_auc}")
        logger.info(f"Classification Report:\n{report}")
        
        # Log metrics to MLflow
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_text(report, "classification_report.txt")
        
        logger.info("Model Evaluation completed successfully.")
    
    except Exception as e:
        logger.error(f"Model Evaluation failed: {e}")
        raise e