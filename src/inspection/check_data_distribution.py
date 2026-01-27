'''
Authors :
- Julian van Zwol
- Sohrab Hakimi
- Roel van Eeten

this file contains functions to check dataset class distribution

how to use:
1. check_dataset_distribution(dataset, dataset_name)
    - dataset: Keras dataset to analyze
    - dataset_name: optional name for display purposes
    - Prints the number of images per class

'''
def check_dataset_distribution(dataset, dataset_name="Dataset"):
    # Haal de klassenamen op
    class_names = dataset.class_names
    
    # Maak een lijst van alle labels in de dataset
    all_labels = []
    for _, labels in dataset:
        all_labels.extend(labels.numpy())
    
    # Tel de voorkomens
    print(f"\n--- Verdeling voor {dataset_name} ---")
    for i, class_name in enumerate(class_names):
        count = all_labels.count(i)
        print(f"{class_name}: {count} afbeeldingen")