import tensorflow as tf
import numpy as np
import numpy as np
import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def deep_learning(train_ds, val_ds, image_size, data_augmentation_layer):
    num_classes = len(train_ds.class_names)
    input_shape = image_size + (3,)

    model = keras.Sequential([
        keras.Input(shape=input_shape),
        data_augmentation_layer,
        keras.layers.Rescaling(1./255),
        keras.layers.Normalization(axis=-1), 

        # 1. CONVOLUTIONAL LAAG: Leert features te extraheren
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 2. Extra CONVOLUTIONAL LAAG
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        # 3. De output "platte" maken (Flatten)
        keras.layers.Flatten(),
        
        # 4. DENSE LAAG (Klassieke Neurale Netwerk): Voor classificatie
        keras.layers.Dense(128, activation='relu'),
        
        # 5. OUTPUT LAAG: Aantal eenheden gelijk aan het aantal klassen
        keras.layers.Dense(num_classes, activation='softmax') # 'softmax' voor multi-klasse classificatie
    ])

    model.summary()

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Gebruik SC_Crossentropy voor integer labels
                metrics=['accuracy'])
    
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20
    )

    # --- Learning curves ---
    plt.figure(figsize=(10,5))
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    val_loss, val_acc = model.evaluate(val_ds)
    print("Validation accuracy:", val_acc)


    # Convert val_ds to numpy arrays for predictions
    X_val = []
    y_val = []

    for batch_imgs, batch_labels in val_ds:
        X_val.append(batch_imgs.numpy())
        y_val.append(batch_labels.numpy())

    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)

    pred_probs = model.predict(X_val)
    y_pred = np.argmax(pred_probs, axis=1)

    cm = confusion_matrix(y_val, y_pred, normalize='true')

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap="Blues",
                xticklabels=train_ds.class_names,
                yticklabels=train_ds.class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.show()