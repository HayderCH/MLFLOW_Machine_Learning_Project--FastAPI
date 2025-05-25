from data import load_data, split_data, preprocess_data
from model import train_model, evaluate_model, save_model
import mlflow
import mlflow.sklearn


def main():
    # Load and split data
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)

    # MLflow tracking
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("random_state", 42)
        mlflow.log_param("scaler", "StandardScaler")

        # Train model
        model = train_model(X_train_scaled, y_train)

        # Evaluate model
        acc, report, conf_matrix = evaluate_model(model, X_test_scaled, y_test)
        mlflow.log_metric("accuracy", acc)

        print(f"Accuracy: {acc:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)

        # Save model and scaler locally
        save_model(model, scaler)

        # Log model and scaler as artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact("scaler.joblib")


if __name__ == "__main__":
    main()
