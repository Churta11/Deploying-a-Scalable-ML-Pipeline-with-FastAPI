import pytest
from train_model import train_model
from ml.model import compute_model_metrics, save_model, load_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    # Returns trained model
    """
    X_train = np.random.rand(20,5)
    y_train = np.random.randint(0, 2, 20)
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_compute_metrics():
    """
    # Test computed model metrics
    """
    y_true = [1,0,1,1]
    y_preds = [0, 1, 1, 0]
    fbeta, precision, recall = compute_model_metrics(y_true, y_preds)
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_save_load_model(tmp_path):
    """
    # Test creates file, saves file and ensures file loads
    """
    X_train = np.random.rand(20,5)
    y_train = np.random.randint(0, 2, 20)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    model_path = tmp_path / 'model.pkl'

    save_model(model, model_path)

    assert model_path.exists()

    loaded_model = load_model(model_path)
    
    assert isinstance(loaded_model, RandomForestClassifier)
    pass
