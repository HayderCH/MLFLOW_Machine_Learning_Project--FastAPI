import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return acc, report, conf_matrix


def save_model(model, scaler, model_path="model.joblib", scaler_path="scaler.joblib"):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)


def load_model(model_path="model.joblib", scaler_path="scaler.joblib"):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler
