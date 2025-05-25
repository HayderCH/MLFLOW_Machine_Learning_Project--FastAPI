import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(csv_path="parkinsons.csv"):
    df = pd.read_csv(csv_path)
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    X = df.drop(columns=["class"])
    y = df["class"]
    return X, y


def preprocess_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
