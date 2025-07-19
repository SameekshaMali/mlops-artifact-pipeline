import joblib
from sklearn.linear_model import LogisticRegression
from src.utils import load_config, load_data

def train_model(X, y, config):
    model = LogisticRegression(
        C=config["C"],
        solver=config["solver"],
        max_iter=config["max_iter"]
    )
    model.fit(X, y)
    return model

if __name__ == "__main__":
    config = load_config()
    X, y = load_data()
    model = train_model(X, y, config)
    joblib.dump(model, "model_train.pkl")
