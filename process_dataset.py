import os
import pickle
from tqdm import tqdm
from feature_extractor import crop_center, get_feature_vector

def process_and_save_features(dataset_path='Images'):
    """
    Loops through all images in the dataset, extracts their features,
    and saves them to a file with a per-image progress bar.
    """
    
    image_paths = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    all_features = {}
    print(f"Found {len(image_paths)} images to process.")
    print("Starting feature extraction for the dog dataset (face-focused)...")
    
    for image_path in tqdm(image_paths):
        cropped_dog_img = crop_center(image_path)
        
        feature_vector = get_feature_vector(cropped_dog_img)
        
        if feature_vector is not None:
            all_features[image_path] = feature_vector

    with open('dog_features.pkl', 'wb') as f:
        pickle.dump(all_features, f)
        
    print(f"\nFeature extraction complete! Processed {len(all_features)} images.")
    print("New face-focused features saved to 'dog_features.pkl'")

if __name__ == '__main__':
    process_and_save_features()