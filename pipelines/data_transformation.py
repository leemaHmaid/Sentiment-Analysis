from src.data.transformation import preprocess_data
import pandas as pd
import os

def run(config):
    """
    Run the data transformation pipeline.

    Args:
        config (Config): Configuration object.
    """
    input_path = config.get('data', 'ingested_data_path')
    output_path = config.get('data', 'processed_data_path')
    df = pd.read_csv(input_path)
    df = preprocess_data(df)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
