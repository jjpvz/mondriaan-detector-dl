import keras
import numpy as np
from pathlib import Path
import time
import cv2 as cv
from GUI.gui import show_directory_selection_window, show_prediction_window

IMAGE_SIZE = (224, 224)

def predict_single_image(model, image_path, class_names):
    """Laadt een afbeelding en maakt een voorspelling."""
    # Laad en preprocess de afbeelding
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Maak batch dimensie
   # img_array = img_array / 255.0  # Normaliseer
    
    # Voorspelling
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], predicted_class_idx, confidence

def test_model(model, class_names, test_dir=r'C:\Users\engineer1\Desktop\Data\personal\HAN\EVD3\mondriaan-detector-dl\testset_2_label'):
    """Test het model op alle afbeeldingen in een directory."""
    print(f"\n{'='*60}")
    print(f"MODEL TESTEN OP: {test_dir}")
    print(f"{'='*60}\n")
    
    # Verzamel alle afbeeldingen in de directory
    test_path = Path(test_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = [f for f in test_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    print(f"Aantal gevonden afbeeldingen: {len(image_files)}\n")
    print("-" * 60)
    counter_correct = 0
    total_images = len(image_files)

    # Loop door alle afbeeldingen
    for img_path in sorted(image_files):
        start_time = time.time()
        predicted_class, predicted_idx, confidence = predict_single_image(model, img_path, class_names)
        prediction_time = time.time() - start_time
        
        print(f"Afbeelding: {img_path.stem.split(' ')[0]}")
        print(f"Voorspelling: {predicted_class} (index: {predicted_idx}) - Zekerheid: {confidence*100:.2f}%")
        print(f"Tijd: {prediction_time*1000:.2f} ms")
        print("-" * 60)
        if img_path.stem.split(' ')[0] == predicted_class:
            counter_correct += 1

    accuracy = (counter_correct / total_images) * 100 if total_images > 0 else 0
    print(f"\nTotale nauwkeurigheid op testset: {accuracy:.2f}% ({counter_correct} van de {total_images} correct voorspeld)")
    return


def test_model_gui(model):
    files_selected = []
    path_str = show_directory_selection_window()
    if path_str is None:
        print("Geen map geselecteerd. Programma wordt afgesloten.")
        exit(0)
    path = Path(path_str)
    for img in path.glob("*.JPG"):
        files_selected.append(img)

    for img in files_selected:
        frame = cv.imread(str(img))  
        
        if frame is None:
            print(f"Kon afbeelding niet laden: {img}")
            exit(0)
        class_names = ['mondriaan1', 'mondriaan2', 'mondriaan3', 'mondriaan4', 'niet_mondriaan']    
        predicted_class, predicted_idx, confidence = predict_single_image(model, img, class_names)
        
        show_prediction_window(frame, predicted_class, confidence, auto_close_ms=5000)
        print(f"Voorspelling: {predicted_class} (index: {predicted_idx}) - Zekerheid: {confidence*100:.2f}%")
