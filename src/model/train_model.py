'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains the function to train a model

how to use:
1. train_model(model, train_ds, val_ds)
    - model: compiled Keras model to train
    - train_ds: training dataset
    - val_ds: validation dataset
    - Returns: training history object with loss and accuracy metrics

'''
def train_model(model, train_ds, val_ds):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30
    )

    return history