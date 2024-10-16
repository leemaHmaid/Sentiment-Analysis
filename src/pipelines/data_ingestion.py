import os
import requests
import zipfile
from src.utils.logger import setup_logger
from src.utils.config import load_config

def run_data_ingestion():
    # Load configuration
    config = load_config(os.path.join('backend/configs', 'config.yaml'))
    
    # Set up logging
    logger = setup_logger(config['logging']['log_file'], config['logging']['level'])
    
    raw_data_path = config['data']['raw_data_path']
    data_url = 'https://github.com/pmensah28/data/raw/main/IMDB-Dataset.csv.zip'
    
    logger.info("\nStarting Data Ingestion...")
    
    try:
        raw_data_dir = os.path.dirname(raw_data_path)
        os.makedirs(raw_data_dir, exist_ok=True)
        logger.info(f"Ensured that the directory {raw_data_dir} exists.")
        
        # Download the zip file
        logger.info(f"\n\nDownloading data from {data_url}")
        response = requests.get(data_url, stream=True)
        response.raise_for_status()  # Raise an error for bad status codes
        
        zip_filename = os.path.basename(data_url)
        zip_path = os.path.join(raw_data_dir, zip_filename)
        
        logger.info(f"Saving zip file to {zip_path}")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info("\nDownload completed successfully.")
        
        # Extract the zip file
        logger.info(f"Extracting {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(raw_data_dir)
        logger.info("\nExtraction completed successfully.")
        
        # Remove the zip file after extraction
        os.remove(zip_path)
        logger.info(f"Removed the zip file {zip_path}")
        
        # Verify that the extracted CSV exists
        extracted_csv = os.path.join(raw_data_dir, 'IMDB-Dataset.csv')
        if os.path.exists(extracted_csv):
            logger.info(f"\nData ingestion successful. File saved to {extracted_csv}")
        else:
            logger.error(f"Expected file {extracted_csv} not found after extraction.")
    
    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request error during data ingestion: {req_err}")
        raise req_err
    except zipfile.BadZipFile as zip_err:
        logger.error(f"Bad zip file error: {zip_err}")
        raise zip_err
    except Exception as e:
        logger.error(f"Data Ingestion failed: {e}")
        raise e

if __name__ == "__main__":
    run_data_ingestion()
