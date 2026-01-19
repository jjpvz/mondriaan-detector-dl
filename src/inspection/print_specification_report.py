from sklearn.metrics import classification_report, f1_score
import numpy as np

def print_specification_report(y_val, y_pred, train_ds):
    print("\n--- Model Performance Report ---")
    # Gebruik labels parameter om alle klassen te specificeren
    labels = np.arange(len(train_ds.class_names))
    report = classification_report(y_val, y_pred, labels=labels, target_names=train_ds.class_names, zero_division=0)
    print(report)

    f1 = f1_score(y_val, y_pred, labels=labels, average='weighted', zero_division=0)
    print(f"Gemiddelde F1-score: {f1:.4f}")