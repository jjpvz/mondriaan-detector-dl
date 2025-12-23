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