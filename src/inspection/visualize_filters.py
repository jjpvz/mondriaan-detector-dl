'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to visualize convolutional filters

how to use:
1. visualize_filters(model)
    - model: trained Keras CNN model
    - Prints filter shapes for all Conv2D layers
    - Visualizes the first 6 filters from the first Conv2D layer

'''
import matplotlib.pyplot as plt
import keras

def visualize_filters(model):
    # summarize filter shapes
    for layer in model.layers:
        # check for convolutional layer
        if not isinstance(layer, keras.layers.Conv2D):
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(layer.name, filters.shape)

    # Example of visualizing some filters
    filters, biases = model.get_layer("conv2d").get_weights()

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    plt.figure(figsize=(6, 8))
    n_filters, ix = 6, 1

    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(f[:, :, j], cmap='gray')
            ix += 1

    plt.suptitle("Learned filters (conv2d)")
    plt.show()