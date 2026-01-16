import keras

from inspection.print_specification_report import print_specification_report
from preprocessing.augment_images import augment_images, display_augmented_images
from preprocessing.enhance_images import display_enhanced_images, enhance_images

from model.find_optimal_model import find_optimal_model
from model.get_optimal_model import get_optimal_model
from model.train_model import train_model
from model.test_model import test_model, test_model_gui

from inspection.plot_confusion_matrix import plot_confusion_matrix
from inspection.print_validation_accuracy import print_validation_accuracy
from inspection.visualize_featuremaps import visualize_featuremaps
from inspection.visualize_filters import visualize_filters
from inspection.get_predictions import get_predictions
from inspection.plot_learning_curves import plot_learning_curves

image_size = (224, 224)
batch_size = 128

if __name__ == "__main__":
   
   '''
   
   train_ds, val_ds = keras.utils.image_dataset_from_directory(
        r"C:\GIT\mondriaan-detector-dl\data\fullset",
        validation_split=0.2,
        subset="both",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    # DEBUG
    #check_dataset_distribution(train_ds, "Training Set")
    #check_dataset_distribution(val_ds, "Validation Set")
    
    # DEBUG
   # display_augmented_images(train_ds)
   # display_enhanced_images(train_ds)

    # Create additional layers
   augmentation = augment_images()
    #enhancement = enhance_images() --> Not needed, provides no improvements

   input_shape = image_size + (3,)
   num_classes = len(train_ds.class_names)

   find_optimal_model(input_shape, num_classes, train_ds, val_ds, augmentation) # --> Outcome defined in ./get_optimal_model.py
   model = get_optimal_model(input_shape, num_classes, augmentation)
   history = train_model(model, train_ds, val_ds)
   y_val, y_pred = get_predictions(model, val_ds)

   print_specification_report(y_val, y_pred, train_ds)
  #  visualize_filters(model)
  #  visualize_featuremaps(model, val_ds)
   plot_learning_curves(history)
   print_validation_accuracy(model, val_ds)
   plot_confusion_matrix(y_val, y_pred, train_ds)
   
   #uncomment this to test the model
   '''
    # Laad het model en test op testset
   model_loaded = keras.models.load_model(r"C:\GIT\mondriaan-detector-dl\results\mondriaan_detector_DL.keras")

    # Definieer de class names
   class_names = ['mondriaan1', 'mondriaan2', 'mondriaan3', 'mondriaan4', 'niet_mondriaan']
    
    # Test het geladen model op testset
   test_model(model_loaded, class_names, test_dir=r'C:\GIT\mondriaan-detector-dl\data\testset')
   
