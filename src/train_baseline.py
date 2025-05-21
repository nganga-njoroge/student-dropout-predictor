import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from prepare_data import load_and_prepare_data

def plot_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"outputs/{filename}", dpi=300)
    plt.close()

def train_and_evaluate():
    X_train, X_test, y_train, y_test, _, target_le = load_and_prepare_data()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier()
    }

    for name, model in models.items():
        print(f"\n📌 Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n📊 {name} Evaluation:")
        report = classification_report(y_test, y_pred, target_names=target_le.classes_)
        print(report)

        # Save report to file
        report_path = f"outputs/{name.lower().replace(' ', '_')}_report.txt"
        with open(report_path, "w") as f:
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