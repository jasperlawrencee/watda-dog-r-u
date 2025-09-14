import os
import pickle
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
from feature_extractor import crop_human_face, get_feature_vector

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

print("Loading dog features...")
with open('dog_features.pkl', 'rb') as f:
    dog_features = pickle.load(f)
print("Dog features loaded successfully!")

def find_best_match(user_features, all_dog_features):
    best_match_path = None
    min_distance = float('inf')
    for dog_path, dog_vec in all_dog_features.items():
        dist = cosine(user_features, dog_vec)
        if dist < min_distance:
            min_distance = dist
            best_match_path = dog_path
    return best_match_path

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files: return "No file part", 400
        file = request.files['file']
        if file.filename == '': return "No selected file", 400

        if file:
            filename = secure_filename(file.filename)
            user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(user_image_path)
            
            cropped_face_img = crop_human_face(user_image_path)
            
            if cropped_face_img is None:
                return "Could not find a face in the uploaded image. Please try another one!", 400

            user_features = get_feature_vector(cropped_face_img)
            
            best_dog_image_path = find_best_match(user_features, dog_features)
            
            display_user_path = os.path.join('uploads', filename)
            display_dog_path = best_dog_image_path
            
            return render_template('result.html', user_image=display_user_path, result_image=display_dog_path)

    return render_template('index.html')

@app.route('/Images/<path:filename>')
def serve_dog_image(filename):
    return send_from_directory('Images', filename)

@app.route('/uploads/<path:filename>')
def serve_user_image(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    app.run(debug=True)