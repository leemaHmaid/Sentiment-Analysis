import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import os
import shutil
import tempfile
from src.pipelines.data_validation import run_data_validation
from src.pipelines.data_transformation import run_data_transformation
from src.pipelines.model_training import run_model_training
from src.pipelines.model_evaluation import run_model_evaluation

class TestModelEvaluation(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.raw_data_path = os.path.join(self.test_dir, 'raw_data.csv')
        self.processed_data_path = os.path.join(self.test_dir, 'processed')
        os.makedirs(self.processed_data_path, exist_ok=True)

        # Create a minimal raw data file
        df = pd.DataFrame({
            'review': ['This movie was great!'] * 10,
            'sentiment': [1] * 10
        })
        df.to_csv(self.raw_data_path, index=False)

        # Mock configuration
        self.config = {
            'data': {
                'raw_data_path': self.raw_data_path,
                'processed_data_path': self.processed_data_path,
                'external_data_path': os.path.join(self.test_dir, 'external_data_sources.csv')
            },
            'model': {
                'bert_model_name': 'bert-base-uncased',
                'num_classes': 2,
                'max_length': 128,
                'dropout_rate': 0.1,
                'output_dir': os.path.join(self.test_dir, 'models'),
                'mlflow_tracking_uri': 'http://127.0.0.1:8080',  # Update to new port if changed
                'mlflow_experiment_name': 'sentiment_analysis_experiment',
                'mlflow_artifact_uri': os.path.join(self.test_dir, 'mlruns')
            },
            'training': {
                'batch_size': 16,
                'num_epochs': 1,  # Reduced for testing
                'learning_rate': 2e-5,
                'weight_decay': 0.01,
                'logging_steps': 10,
                'save_steps': 1000,
                'evaluation_strategy': 'steps',
                'eval_steps': 500
            },
            'logging': {
                'level': 'INFO',
                'log_file': os.path.join(self.test_dir, 'training.log')
            },
            'storage': {
                's3_bucket': 'your-s3-bucket-name',
                's3_region': 'your-region'
            },
            'validation': {
                'required_columns': ['review', 'sentiment'],
                'missing_value_threshold': 0.05,  # 5%
                'duplicate_subset': ['review']
            }
        }

        # Patch load_config to return the mocked config
        self.patcher_load_config = patch('src.utils.config.load_config', return_value=self.config)
        self.mock_load_config = self.patcher_load_config.start()

        # Patch setup_logger to return a mock logger
        self.patcher_setup_logger = patch('src.utils.logger.setup_logger', return_value=MagicMock())
        self.mock_setup_logger = self.patcher_setup_logger.start()

        # Patch MLflow methods in the context of model_training
        self.patcher_set_experiment = patch('src.pipelines.model_training.mlflow.set_experiment')
        self.mock_set_experiment = self.patcher_set_experiment.start()

        self.patcher_start_run = patch('src.pipelines.model_training.mlflow.start_run')
        self.mock_start_run = self.patcher_start_run.start()

        self.patcher_log_params = patch('src.pipelines.model_training.mlflow.log_params')
        self.mock_log_params = self.patcher_log_params.start()

        self.patcher_log_metrics = patch('src.pipelines.model_training.mlflow.log_metrics')
        self.mock_log_metrics = self.patcher_log_metrics.start()

        self.patcher_log_artifact = patch('src.pipelines.model_training.mlflow.log_artifact')
        self.mock_log_artifact = self.patcher_log_artifact.start()

        self.patcher_log_model = patch('src.pipelines.model_training.mlflow.pytorch.log_model')
        self.mock_log_model = self.patcher_log_model.start()

    def tearDown(self):
        # Remove the temporary directory after the test
        shutil.rmtree(self.test_dir)

        # Stop all patchers
        self.patcher_load_config.stop()
        self.patcher_setup_logger.stop()
        self.patcher_set_experiment.stop()
        self.patcher_start_run.stop()
        self.patcher_log_params.stop()
        self.patcher_log_metrics.stop()
        self.patcher_log_artifact.stop()
        self.patcher_log_model.stop()

    def test_run_model_evaluation_success(self):
        # Run Data Validation
        try:
            run_data_validation()
        except Exception as e:
            self.fail(f"Data Validation failed during model evaluation test: {e}")

        # Run Data Transformation
        try:
            run_data_transformation()
        except Exception as e:
            self.fail(f"Data Transformation failed during model evaluation test: {e}")

        # Run Model Training
        try:
            run_model_training()
        except Exception as e:
            self.fail(f"Model Training failed during model evaluation test: {e}")

        # Run Model Evaluation
        try:
            run_model_evaluation()
        except Exception as e:
            self.fail(f"Model Evaluation failed: {e}")

        # Assertions to ensure MLflow methods were called
        self.mock_set_experiment.assert_called_once_with('sentiment_analysis_experiment')
        self.mock_start_run.assert_called_once()
        self.mock_log_params.assert_called()
        self.mock_log_metrics.assert_called()
        self.mock_log_artifact.assert_called()
        self.mock_log_model.assert_called()

    def test_run_model_evaluation_insufficient_data(self):
        # Modify the dataset to have insufficient data for evaluation
        df = pd.read_csv(self.raw_data_path)
        df = df.sample(n=1)  # Reduce dataset size to 1 sample
        df.to_csv(self.raw_data_path, index=False)

        # Run Data Validation
        try:
            run_data_validation()
        except Exception as e:
            self.fail(f"Data Validation failed during insufficient data evaluation test: {e}")

        # Run Data Transformation
        try:
            run_data_transformation()
        except Exception as e:
            pass

        # Run Model Training
        try:
            run_model_training()
        except Exception as e:
            self.fail(f"Model Training failed with insufficient data during evaluation test: {e}")

        # Run Model Evaluation
        try:
            run_model_evaluation()
        except Exception as e:
            self.fail(f"Model Evaluation failed with insufficient data: {e}")

        # Assertions to ensure MLflow methods were called
        self.mock_set_experiment.assert_called_once_with('sentiment_analysis_experiment')
        self.mock_start_run.assert_called_once()
        self.mock_log_params.assert_called()
        self.mock_log_metrics.assert_called()
        self.mock_log_artifact.assert_called()
        self.mock_log_model.assert_called()

if __name__ == '__main__':
    unittest.main()
