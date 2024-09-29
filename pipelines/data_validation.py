from src.data.validation import validate_data
import pandas as pd

def run(config):
    """
    Run the data validation pipeline.

    Args:
        config (Config): Configuration object.
    """
    input_path = config.get('data', 'ingested_data_path')
    df = pd.read_csv(input_path)
    validate_data(df)
