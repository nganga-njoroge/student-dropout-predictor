import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_data import load_and_prepare_data

# Try importing XGBoost if available
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False

# Ensure outputs/ folder exists
os.makedirs("outputs", exist_ok=True)

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=300)
    plt.close()

def compute_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

def train_and_evaluate():
    X_train, X_test, y_train, y_test, _, target_le = load_and_prepare_data()
    class_weights = compute_weights(y_train)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            class_weight=class_weights,
            random_state=42
        )
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(
            use_label_encoder=False,
            eval_metric="mlogloss",
            scale_pos_weight=class_weights.get(1, 1)
        )

    for name, model in models.items():
        print(f"\n📌 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n📊 {name} Evaluation:")
        report = classification_report(y_test, y_pred, target_names=target_le.classes_)
        print(report)

        # Save report
        with open(f"outputs/{name.lower().replace(' ', '_')}_report.txt", "w") as f:
            f.write(report)

        # Save confusion matrix
        plot_confusion(
            y_test,
            y_pred,
            f"{name} Confusion Matrix",
            f"{name.lower().replace(' ', '_')}_cm.png"
        )

if __name__ == "__main__":
    train_and_evaluate()