from src.data.ingestion import load_data
import os

def run(config):
    """
    Run the data ingestion pipeline.

    Args:
        config (Config): Configuration object.
    """
    file_path = config.get('data', 'raw_data_path')
    df = load_data(file_path)
    output_path = config.get('data', 'ingested_data_path')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
