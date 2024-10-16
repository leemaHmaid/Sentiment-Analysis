import os
import yaml
import pandas as pd
from src.utils.logger import setup_logger
from src.utils.config import load_config

def run_data_validation():
    # Load configuration
    config = load_config(os.path.join('backend/configs', 'config.yaml'))
    
    # Set up logging
    logger = setup_logger(config['logging']['log_file'], config['logging']['level'])
    
    raw_data_path = config['data']['raw_data_path']
    
    logger.info("\n\nStarting Data Validation...")
    
    try:
        # Load raw dataset
        logger.info(f"\nLoading raw data from {raw_data_path} for validation.")
        df = pd.read_csv(raw_data_path)
        
        # Schema Validation
        required_columns = config['validation']['required_columns']
        logger.info(f"Validating required columns: {required_columns}")
        for column in required_columns:
            if column not in df.columns:
                logger.error(f"Missing required column: {column}")
                raise ValueError(f"Missing required column: {column}")
        
        # Missing Values Validation
        missing_value_threshold = config['validation']['missing_value_threshold']
        logger.info(f"Checking for missing values with threshold {missing_value_threshold*100}%")
        missing_values = df.isnull().mean()
        for column, missing_ratio in missing_values.items():
            if missing_ratio > missing_value_threshold:
                logger.error(f"Column '{column}' has {missing_ratio*100:.2f}% missing values, which exceeds the threshold.")
                raise ValueError(f"Column '{column}' has {missing_ratio*100:.2f}% missing values.")
            else:
                logger.info(f"Column '{column}' has acceptable missing values: {missing_ratio*100:.2f}%")
        
        # Duplicate Records Validation
        duplicate_subset = config['validation']['duplicate_subset']
        initial_count = df.shape[0]
        df.drop_duplicates(subset=duplicate_subset, inplace=True)
        final_count = df.shape[0]
        duplicates_removed = initial_count - final_count
        logger.info(f"Removed {duplicates_removed} duplicate records based on subset {duplicate_subset}.")
        
        if 'review_length' not in df.columns:
            df['review_length'] = df['review'].apply(lambda x: len(str(x)))
        
        review_length_mean = df['review_length'].mean()
        review_length_std = df['review_length'].std()
        upper_bound = review_length_mean + 3 * review_length_std
        lower_bound = review_length_mean - 3 * review_length_std
        outliers = df[(df['review_length'] > upper_bound) | (df['review_length'] < lower_bound)]
        logger.info(f"Identified {outliers.shape[0]} outlier records based on review length.")
                
        logger.info("\nData Validation completed successfully.")
    
    except Exception as e:
        logger.error(f"Data Validation failed: {e}")
        raise e
