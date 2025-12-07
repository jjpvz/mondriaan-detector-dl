import keras
import tensorflow as tf
from deep_learning import deep_learning
from augment_images import augment_images, display_augmented_images
from enhance_images import display_enhanced_images, enhance_images

image_size = (224, 224)
batch_size = 128

if __name__ == "__main__":
    train_ds, val_ds = keras.utils.image_dataset_from_directory(
        r"C:\workspace\evml\EVD3\mondriaan-detector-dl\data\fullset",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    
    # Debug
    # display_augmented_images(train_ds)
    # display_enhanced_images(train_ds)

    # Create additional layers
    # augmentation = augment_images()
    # enhancement = enhance_images()

    # deep_learning(train_ds, val_ds, image_size, augmentation)