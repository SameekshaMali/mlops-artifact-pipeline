import pytest
from src.train import train_model
from src.utils import load_config, load_data
from sklearn.linear_model import LogisticRegression

def test_config_file_loading():
    config = load_config()
    assert isinstance(config["C"], float)
    assert isinstance(config["solver"], str)
    assert isinstance(config["max_iter"], int)

def test_model_creation():
    X, y = load_data()
    config = load_config()
    model = train_model(X, y, config)
    assert isinstance(model, LogisticRegression)

def test_model_accuracy():
    X, y = load_data()
    config = load_config()
    model = train_model(X, y, config)
    acc = model.score(X, y)
    assert acc > 0.8
