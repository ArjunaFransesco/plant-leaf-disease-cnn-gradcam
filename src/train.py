"""
Model Training & Evaluation Script
Author: Arjuna Fransesco
"""

import json
import os
import sys

# Ensure root directory is on python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from src.model import LeafDiseaseCNNClassifier, FEATURE_COLS
from src.dataset import DISEASE_CLASSES

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


def train_and_evaluate():
    print("[*] Loading Kaggle Leaf Pathology Dataset...")
    df = pd.read_csv("data/dataset_metadata.csv")
    
    train_df = df[df["split"].isin(["train", "val"])]
    test_df = df[df["split"] == "test"]

    X_train, y_train = train_df[FEATURE_COLS], train_df["label"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["label"]

    print(f"[*] Training on {len(X_train)} samples, testing on {len(X_test)} samples...")
    model = LeafDiseaseCNNClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))
    macro_prec = float(precision_score(y_test, preds, average="macro"))
    macro_rec = float(recall_score(y_test, preds, average="macro"))

    print(f"[+] Test Accuracy: {acc * 100:.2f}%")
    print(f"[+] Macro F1-Score: {macro_f1:.4f}")

    # Save model
    model.save("models/plant_leaf_disease_model.joblib")
    print("[+] Model saved to models/plant_leaf_disease_model.joblib")

    # Metrics JSON
    metrics = {
        "Model_Architecture": "Deep Feature Extractor + Gradient Boosted Ensemble",
        "Test_Accuracy_Pct": round(acc * 100, 2),
        "Macro_F1_Score": round(macro_f1, 4),
        "Macro_Precision": round(macro_prec, 4),
        "Macro_Recall": round(macro_rec, 4),
        "Total_Dataset_Samples": len(df),
        "Test_Samples": len(test_df),
        "Target_Classes": DISEASE_CLASSES
    }
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion Matrix Visualization
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    cm = confusion_matrix(y_test, preds, labels=DISEASE_CLASSES)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="YlGnBu", xticklabels=DISEASE_CLASSES, yticklabels=DISEASE_CLASSES, ax=ax)
    plt.title(f"Plant Leaf Disease Diagnosis Confusion Matrix (Acc = {acc * 100:.1f}%)", fontweight="bold", fontsize=12)
    plt.xlabel("Predicted Class", fontweight="bold")
    plt.ylabel("Ground Truth Class", fontweight="bold")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=200)
    plt.close()
    print("[+] Saved confusion matrix to reports/confusion_matrix.png")


if __name__ == "__main__":
    train_and_evaluate()
