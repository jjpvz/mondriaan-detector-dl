from sklearn.metrics import classification_report, f1_score

def print_specification_report(y_val, y_pred, train_ds):
    print("\n--- Model Performance Report ---")
    report = classification_report(y_val, y_pred, target_names=train_ds.class_names)
    print(report)

    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f"Gemiddelde F1-score: {f1:.4f}")