import unittest
from src.pipelines.training_pipeline import run as training_run
from src.utils.config import Config

class TestTrainingPipeline(unittest.TestCase):
    def test_training_pipeline(self):
        config = Config('config/config.yaml')
        try:
            training_run(config)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Training pipeline failed with exception: {e}")

if __name__ == '__main__':
    unittest.main()
