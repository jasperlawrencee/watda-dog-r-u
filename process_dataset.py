import os
import pickle
from tqdm import tqdm
from feature_extractor import crop_center, get_feature_vector

def process_and_save_features(dataset_path='Images'):
    all_features = {}
    print("Starting feature extraction for the dog dataset (face-focused)...")

    for root, dirs, files in tqdm(os.walk(dataset_path)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)

                # 1. Crop the center of the dog image
                cropped_dog_img = crop_center(image_path)

                # 2. Get features from the cropped image
                feature_vector = get_feature_vector(cropped_dog_img)

                if feature_vector is not None:
                    all_features[image_path] = feature_vector

    with open('dog_features.pkl', 'wb') as f:
        pickle.dump(all_features, f)

    print(f"\nFeature extraction complete! Processed {len(all_features)} images.")
    print("New face-focused features saved to 'dog_features.pkl'")

if __name__ == '__main__':
    process_and_save_features()