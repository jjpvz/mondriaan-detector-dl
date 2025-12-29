import keras
import numpy as np
from pathlib import Path

IMAGE_SIZE = (224, 224)

def predict_single_image(model, image_path, class_names):
    """Laadt een afbeelding en maakt een voorspelling."""
    # Laad en preprocess de afbeelding
    img = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Maak batch dimensie
    img_array = img_array / 255.0  # Normaliseer
    
    # Voorspelling
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_idx]
    
    return class_names[predicted_class_idx], predicted_class_idx, confidence

def test_model(model, class_names, test_dir=r'C:\GIT\mondriaan-detector-dl\data\testset'):
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
    
    # Loop door alle afbeeldingen
    for img_path in sorted(image_files):
        predicted_class, predicted_idx, confidence = predict_single_image(model, img_path, class_names)
        print(f"Afbeelding: {img_path.name}")
        print(f"Voorspelling: {predicted_class} (index: {predicted_idx}) - Zekerheid: {confidence*100:.2f}%")
        print("-" * 60)

