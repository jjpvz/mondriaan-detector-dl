'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains the optimal CNN model architecture

how to use:
1. get_optimal_model(input_shape, num_classes, data_augmentation_layer)
    - input_shape: tuple of image dimensions (height, width, channels)
    - num_classes: number of output classes
    - data_augmentation_layer: augmentation layer to apply
    - Returns: compiled Keras Sequential model with 3 Conv2D layers
    - Architecture: Conv2D -> MaxPooling (x3) -> Dense -> Softmax

'''
import tensorflow as tf
import keras

def get_optimal_model(input_shape, num_classes, data_augmentation_layer):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        data_augmentation_layer,
        keras.layers.Rescaling(1./255),
        keras.layers.Normalization(axis=-1), 

        keras.layers.Conv2D(24, (7, 7), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        
        keras.layers.Conv2D(48, (7, 7), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (7, 7), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # Gebruik SC_Crossentropy voor integer labels
                metrics=['accuracy'])
    
    return model