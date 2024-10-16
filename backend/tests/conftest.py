import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

@pytest.fixture(scope='function', autouse=True)
def mock_model_loading(mocker):
    """
    Mock the model loading to prevent FileNotFoundError.
    This fixture runs automatically for all tests.
    """
    mocker.patch('torch.load', return_value=MagicMock())

    mocker.patch(
        'transformers.BertForSequenceClassification.from_pretrained',
        return_value=MagicMock()
    )

    # Create a mocked model instance
    mocked_model = MagicMock()
    mocked_model.load_state_dict = MagicMock(return_value=None)
    mocked_model.eval = MagicMock()
    
    # Patch the model in the app with the mocked model
    mocker.patch('app.app.model', mocked_model)

    # Import the app after mocking to ensure mocks are in place
    from app.app import app
    client = TestClient(app)
    return client

@pytest.fixture(scope='function')
def client(mock_model_loading):
    """
    Provide the TestClient with the mocked app.
    """
    return mock_model_loading
