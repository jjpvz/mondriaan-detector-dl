'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to visualize feature maps from convolutional layers

how to use:
1. visualize_featuremaps(model, val_ds)
    - model: trained Keras CNN model
    - val_ds: validation dataset
    - Visualizes feature maps from each Conv2D and MaxPooling layer
    - Shows how the network processes an input image

'''
import keras
import matplotlib.pyplot as plt
import numpy as np

def visualize_featuremaps(model, val_ds):
    ixs = [
        model.layers.index(model.get_layer("conv2d")),
        model.layers.index(model.get_layer("max_pooling2d")),
        model.layers.index(model.get_layer("conv2d_1")),
        model.layers.index(model.get_layer("max_pooling2d_1")),
        model.layers.index(model.get_layer("conv2d_2")),
        model.layers.index(model.get_layer("max_pooling2d_2"))
    ]

    outputs = [model.layers[i].output for i in ixs]

    model_steps = keras.models.Model(
        inputs=model.inputs,
        outputs=outputs
    )

    for images, labels in val_ds.take(1):
        img = images[0:1]

    feature_maps = model_steps.predict(img)

    for fmap in feature_maps:
        n_features = fmap.shape[-1]
        square = int(np.sqrt(n_features))
        ix = 1

        plt.figure(figsize=(8, 8))
        for _ in range(square):
            for _ in range(square):
                if ix > n_features:
                    break
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(fmap[0, :, :, ix - 1], cmap='gray')
                ix += 1

        plt.show()