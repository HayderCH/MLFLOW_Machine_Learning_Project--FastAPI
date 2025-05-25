from data import load_data, split_data, preprocess_data
from model import train_model, evaluate_model


def test_train_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    model = train_model(X_train_scaled, y_train)
    acc, _, _ = evaluate_model(model, X_test_scaled, y_test)
    # Check that model reaches at least 70% accuracy
    assert acc > 0.7
