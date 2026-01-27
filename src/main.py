
from model.transfer_learning_model import transfer_learning_model, finetune_mobilenetv2
from preprocessing.augment_images import augment_images
import tensorflow as tf
import configparser
import keras
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from inspection.print_specification_report import print_specification_report

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

# Set to true when using transfer learning
# Set to false when using regular cnn
transfer_learning = False
image_size = (224, 224)
batch_size = 32  # Reduced from 128 for better training

if __name__ == "__main__":
    # config = configparser.ConfigParser()
    # config.read('config.ini')
    # fullset = config['General']['fullset_evd3_path']
    # model_path = config['General']['model_path']

    # train_ds, val_ds = keras.utils.image_dataset_from_directory(
    #     fr"{fullset}",
    #     validation_split=0.2,
    #     subset="both",
    #     seed=1337,
    #     image_size=image_size,
    #     batch_size=batch_size,
    # )

    # augmentation = augment_images()

    # input_shape = image_size + (3,)
    # num_classes = len(train_ds.class_names)

    # # Calculate class weights for imbalanced dataset
    # class_labels = np.concatenate([y for x, y in train_ds], axis=0)
    # class_weights_array = compute_class_weight(
    #     class_weight='balanced',
    #     classes=np.unique(class_labels),
    #     y=class_labels
    # )
    # class_weights = dict(enumerate(class_weights_array))
    # print(f"Class weights: {class_weights}")

    # if (transfer_learning == False):
    #     # find_optimal_model(input_shape, num_classes, train_ds, val_ds, augmentation) # --> Outcome defined in ./get_optimal_model.py
    #     model = get_optimal_model(input_shape, num_classes, augmentation)
    
    # if (transfer_learning):
    #     model = transfer_learning_model(
    #         input_shape=input_shape,
    #         num_classes=num_classes,
    #         augmentation=augmentation,
    #         dense_units=256,
    #         dropout=0.3,
    #         lr=1e-4
    #     )

    # if (transfer_learning == False):
    #     history = train_model(model, train_ds, val_ds)
    #     model.save(f"{model_path}/mondriaan_detector_DL.keras")

    # if (transfer_learning):
    #     # Phase 1: Train only the top layers first
    #     print("\n=== Phase 1: Training top layers ===")
    #     callbacks_initial = [
    #         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    #     ]
    #     history_initial = model.fit(
    #         train_ds, 
    #         validation_data=val_ds, 
    #         epochs=20,
    #         callbacks=callbacks_initial,
    #         class_weight=class_weights
    #     )
        
    #     # Phase 2: Fine-tune the base model
    #     print("\n=== Phase 2: Fine-tuning base model ===")
    #     model = finetune_mobilenetv2(model, fine_tune_at=120, lr=1e-5)
    #     callbacks_ft = [
    #         tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    #         tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    #     ]
    #     history_ft = model.fit(
    #         train_ds, 
    #         validation_data=val_ds, 
    #         epochs=30,
    #         callbacks=callbacks_ft,
    #         class_weight=class_weights
    #     )
    #     model.save(f"{model_path}/mondriaan_detector_DL_transferLearning.keras")

    # y_val, y_pred = get_predictions(model, val_ds)

    # # Determine which inspection methods to use by (un)commenting:
    # print_specification_report(y_val, y_pred, train_ds)
    # if not transfer_learning:
    #     plot_learning_curves(history)
    # plot_confusion_matrix(y_val, y_pred, train_ds)

    # if transfer_learning:
    #     # Combine both training phases for complete learning curves
    #     combined_history = {}
    #     for key in history_initial.history.keys():
    #         combined_history[key] = history_initial.history[key] + history_ft.history[key]
        
    #     class CombinedHistory:
    #         def __init__(self, history_dict):
    #             self.history = history_dict
        
#         plot_learning_curves(CombinedHistory(combined_history))
    # visualize_filters(model)
    # visualize_featuremaps(model, val_ds)
    # print_validation_accuracy(model, val_ds)
   
    # # TODO: What to do with this?
    # #uncomment this to test the model
    # # Laad het model en test op testset
    model_loaded = keras.models.load_model(r"C:\GIT\mondriaan-detector-dl\models\mondriaan_detector_DL_transferLearning.keras")

    # # Definieer de class names
    # class_names = ['mondriaan1', 'mondriaan2', 'mondriaan3', 'mondriaan4', 'niet_mondriaan']

    # # Test het geladen model op testset
    test_model_gui(model_loaded)