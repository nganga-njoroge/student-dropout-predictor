import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_data(path="data/student_dropout.csv"):
    df = pd.read_csv(path)

    # Optional: map target if not yet string
    if df["Target"].dtype != "object":
        target_map = {0: "Enrolled", 1: "Dropout", 2: "Graduate"}
        df["Target"] = df["Target"].map(target_map)

    # Encode categorical columns (except target)
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        if col != "Target":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Encode target separately
    target_le = LabelEncoder()
    df["Target"] = target_le.fit_transform(df["Target"])

    X = df.drop("Target", axis=1)
    y = df["Target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test, label_encoders, target_le