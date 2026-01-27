'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions for image enhancement

how to use:
1. enhance_images()
    - Returns: Sequential model with enhancement layers
    - Applies: contrast and brightness adjustments
2. display_enhanced_images(dataset)
    - dataset: Keras dataset to enhance
    - Displays original and enhanced versions side by side

'''
import matplotlib.pyplot as plt
import tensorflow as tf
import keras

def enhance_images():
    return keras.Sequential([
        keras.layers.Lambda(lambda x: tf.image.adjust_contrast(x, 1.15)),
        keras.layers.Lambda(lambda x: tf.image.adjust_brightness(x, 0.05))
    ])

def display_enhanced_images(dataset):
    # Pak één batch uit de dataset
    for images, labels in dataset.take(1):
        sample_images = images[:5]  # neem een paar voorbeelden
        break

    # Pas de enhancement layer toe
    enhancement_layer = enhance_images()

    enhanced = enhancement_layer(sample_images) 

    # Plot origineel vs enhanced
    plt.figure(figsize=(12, 6))

    for i in range(5):
        # Origineel
        plt.subplot(2, 5, i + 1)
        plt.imshow(sample_images[i].numpy().astype("uint8"))
        plt.title("Origineel")
        plt.axis("off")

        # Ge-enhanced
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(enhanced[i].numpy().astype("uint8"))
        plt.title("Enhanced")
        plt.axis("off")

    plt.show()
