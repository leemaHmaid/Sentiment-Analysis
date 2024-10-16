import os
import yaml
import pandas as pd
import torch
import warnings
import sys
import time
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from src.utils.logger import setup_logger
from src.utils.config import load_config
import mlflow
from torch.optim import AdamW
import mlflow.pytorch
from tqdm.auto import tqdm

class IMDBDataset(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_length):
        self.reviews = reviews
        self.sentiments = sentiments
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review = str(self.reviews[idx])
        sentiment = self.sentiments[idx]

        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(sentiment, dtype=torch.long)
        }

def run_model_training():
    # Load configuration
    config = load_config(os.path.join('backend/configs', 'config.yaml'))

    # Set up logging
    logger = setup_logger(config['logging']['log_file'], config['logging']['level'])
    
    # Initialize MLflow
    mlflow.set_tracking_uri(config['model']['mlflow_tracking_uri'])
    mlflow.set_experiment(config['model']['mlflow_experiment_name'])
    
    with mlflow.start_run():
        # Log configuration parameters
        mlflow.log_params({
            'bert_model_name': config['model']['bert_model_name'],
            'num_classes': config['model']['num_classes'],
            'max_length': config['model']['max_length'],
            'dropout_rate': config['model']['dropout_rate'],
            'batch_size': config['training']['batch_size'],
            'num_epochs': config['training']['num_epochs'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'logging_steps': config['training']['logging_steps'],
            'save_steps': config['training']['save_steps'],
            'evaluation_strategy': config['training']['evaluation_strategy'],
            'eval_steps': config['training']['eval_steps']
        })
        
        raw_data_path = config['data']['raw_data_path']
        processed_data_path = config['data']['processed_data_path']
        train_data_path = os.path.join(processed_data_path, 'train.csv')
        validation_data_path = os.path.join(processed_data_path, 'validation.csv')
        
        # Load training and validation data
        logger.info("\nLoading training data.")
        train_df = pd.read_csv(train_data_path)
        # Convert sentiments to numeric
        train_df['sentiment'] = train_df['sentiment'].map({'positive': 1, 'negative': 0})

        logger.info("\nLoading validation data.")
        validation_df = pd.read_csv(validation_data_path)
        # Convert sentiments to numeric
        validation_df['sentiment'] = validation_df['sentiment'].map({'positive': 1, 'negative': 0})

        # Initialize tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(config['model']['bert_model_name'])
        model = BertForSequenceClassification.from_pretrained(
            config['model']['bert_model_name'],
            num_labels=config['model']['num_classes'],
            hidden_dropout_prob=config['model']['dropout_rate']
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create datasets and dataloaders
        train_dataset = IMDBDataset(
            reviews=train_df['review'],
            sentiments=train_df['sentiment'],
            tokenizer=tokenizer,
            max_length=config['model']['max_length']
        )
        
        validation_dataset = IMDBDataset(
            reviews=validation_df['review'],
            sentiments=validation_df['sentiment'],
            tokenizer=tokenizer,
            max_length=config['model']['max_length']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=config['training']['batch_size'], shuffle=False)
        
        # Define optimizer and scheduler
        learning_rate = float(config['training']['learning_rate'])
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=config['training']['weight_decay'])
        
        total_steps = len(train_loader) * config['training']['num_epochs']
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        logger.info("\n\nStarting training...")
        for epoch in range(config['training']['num_epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
            logger.info("-" * 10)
            
            # Training phase
            model.train()
            total_train_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = total_train_loss / len(train_loader)
            logger.info(f"Average Training Loss: {avg_train_loss:.4f}")
            mlflow.log_metric("train_loss_epoch", avg_train_loss, step=epoch)
            
            # Validation phase
            model.eval()
            total_eval_loss = 0
            predictions = []
            true_labels = []
            
            progress_bar = tqdm(validation_loader, desc=f"Validation Epoch {epoch + 1}")
            with torch.no_grad():
                for batch in progress_bar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_eval_loss += loss.item()
                    
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).flatten()
                    
                    predictions.extend(preds.cpu().numpy())
                    true_labels.extend(labels.cpu().numpy())
                    
                    progress_bar.set_postfix({'val_loss': loss.item()})
            
            avg_val_loss = total_eval_loss / len(validation_loader)
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
            
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Validation Accuracy: {accuracy:.4f}")
            logger.info(f"Validation Precision: {precision:.4f}")
            logger.info(f"Validation Recall: {recall:.4f}")
            logger.info(f"Validation F1 Score: {f1:.4f}")
            
            mlflow.log_metrics({
                "val_loss_epoch": avg_val_loss,
                "val_accuracy": accuracy,
                "val_precision": precision,
                "val_recall": recall,
                "val_f1": f1
            }, step=epoch)
        
        # Save the trained model
        model_output_dir = config['model']['output_dir']
        os.makedirs(model_output_dir, exist_ok=True)
        model_save_path = os.path.join(model_output_dir, 'bert_classifier.pth')
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Model saved to {model_save_path}")
        mlflow.log_artifact(model_save_path, artifact_path="model_artifacts")
        
        # Log the model with MLflow
        mlflow.pytorch.log_model(model, artifact_path="model_mlflow", registered_model_name="BertClassifierModel")
        
        logger.info("\nModel Training completed successfully!")