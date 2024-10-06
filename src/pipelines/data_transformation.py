import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.logger import setup_logger
from src.utils.config import load_config
import warnings
warnings.filterwarnings("ignore")

def run_data_transformation():
    # Load configuration
    config = load_config(os.path.join('configs', 'config.yaml'))
    
    # Set up logging
    logger = setup_logger(config['logging']['log_file'], config['logging']['level'])
    
    raw_data_path = config['data']['raw_data_path']
    processed_data_path = config['data']['processed_data_path']
    
    logger.info("Starting Data Transformation...")
    
    try:
        # Load validated raw data
        logger.info(f"Loading validated raw data from {raw_data_path}")
        df = pd.read_csv(raw_data_path)
        
        logger.info("Computing review length.")
        df['review_length'] = df['review'].apply(lambda x: len(str(x)))
        
        logger.info("Lowercasing review texts.")
        df['review'] = df['review'].str.lower()
        
        # Split data into train, validation, test
        logger.info("Splitting data into train, validation, and test sets.")
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['sentiment'])
        validation_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['sentiment'])
        
        os.makedirs(processed_data_path, exist_ok=True)
        # Save split datasets
        train_path = os.path.join(processed_data_path, 'train.csv')
        validation_path = os.path.join(processed_data_path, 'validation.csv')
        test_path = os.path.join(processed_data_path, 'test.csv')
        
        logger.info(f"Saving train data to {train_path}")
        train_df.to_csv(train_path, index=False)
        
        logger.info(f"Saving validation data to {validation_path}")
        validation_df.to_csv(validation_path, index=False)
        
        logger.info(f"Saving test data to {test_path}")
        test_df.to_csv(test_path, index=False)
        
        logger.info("Data Transformation completed successfully.")
    
    except Exception as e:
        logger.error(f"Data Transformation failed: {e}")
        raise e
