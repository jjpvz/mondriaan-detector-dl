def print_validation_accuracy(model, val_ds):
    val_loss, val_acc = model.evaluate(val_ds)
    print("Validation accuracy:", val_acc)