import joblib
from src.utils import load_data

if __name__ == "__main__":
    model = joblib.load("model_train.pkl")
    X, y = load_data()
    preds = model.predict(X)
    print("First 10 Predictions:", preds[:10])
