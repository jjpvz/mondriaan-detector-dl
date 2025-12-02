from createModel import create_cnn_model
from displayImage import display_image
from loadData import load_images
from prepareData import prepare_data_for_dl
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    images = load_images("fullset")

    X, y, class_names = prepare_data_for_dl(images)

    input_shape = X.shape[1:] 
    num_classes = len(class_names)

    print(f"Totale dataset vorm (X): {X.shape}") 
    print(f"Aantal klassen: {num_classes}")

    model = create_cnn_model(input_shape, num_classes)

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Gebruik SC_Crossentropy voor integer labels
                metrics=['accuracy'])

    model.summary()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    history = model.fit(
        X_train, 
        y_train, 
        epochs=20,         
        validation_split=0.1,
        shuffle=True)
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'\nTest Nauwkeurigheid: {test_acc:.4f}')

    predictions = model.predict(X_test)
    most_probable_predictions = np.argmax(predictions, axis=1)

    print("\nVoorbeeld van voorspellingen op de testset:")
    for i in range(5):
        print(f"Werkelijke label: {class_names[y_test[i]]} (Index: {y_test[i]})")
        print(f"Voorspeld label: {class_names[most_probable_predictions[i]]} (Index: {most_probable_predictions[i]})\n")