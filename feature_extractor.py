import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model

# --- MODEL INITIALIZATION ---
# We load the model once here to be efficient
base_model = VGG16(weights='imagenet', include_top=False)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# --- CROPPING FUNCTIONS ---

def crop_human_face(image_path):
    """Finds and crops the human face in an image."""
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None # No face found
            
        # Get coordinates of the first face found
        x, y, w, h = faces[0]
        cropped_face = img[y:y+h, x:x+w]
        
        # Convert from OpenCV's BGR format to PIL's RGB format
        cropped_face_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        return Image.fromarray(cropped_face_rgb)
    except Exception as e:
        print(f"Error cropping human face from {image_path}: {e}")
        return None

def crop_center(image_path):
    """Crops the center square of a dog image."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        new_dim = min(width, height)
        
        left = (width - new_dim) / 2
        top = (height - new_dim) / 2
        right = (width + new_dim) / 2
        bottom = (height + new_dim) / 2
        
        return img.crop((left, top, right, bottom))
    except Exception as e:
        print(f"Error center-cropping {image_path}: {e}")
        return None

# --- FEATURE EXTRACTION ---

def get_feature_vector(img_obj):
    """
    This function now takes a PIL Image object and returns its feature vector.
    """
    if img_obj is None:
        return None

    # Resize image to the required 224x224 for VGG16
    img_resized = img_obj.resize((224, 224))
    
    # Convert image to a numpy array
    img_array = np.array(img_resized)
    
    # Ensure it's 3-channel RGB
    if img_array.ndim == 2: # Grayscale
        img_array = np.stack([img_array]*3, axis=-1)
    elif img_array.shape[2] == 4: # RGBA
        img_array = img_array[:,:,:3]
        
    # Add batch dimension and preprocess
    expanded_img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(expanded_img_array)

    # Get the feature vector
    feature_vector = model.predict(processed_img, verbose=0)
    flattened_vector = feature_vector.flatten()
    normalized_vector = flattened_vector / np.linalg.norm(flattened_vector)
    
    return normalized_vector