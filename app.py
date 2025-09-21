import os
import pickle
from flask import Flask, render_template, request, send_from_directory, session
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
from feature_extractor import crop_human_face, get_feature_vector

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for session
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_breed_name(image_path):
    """Extract breed name from image path using folder name pattern n{number}-{breed_name}"""
    parts = image_path.split(os.sep)
    for part in parts:
        if part.startswith('n') and '-' in part:
            breed_name = part.split('-', 1)[1]  # Split at first hyphen
            return breed_name.replace('_', ' ').title()  # Convert "Irish_setter" to "Irish Setter"
    return "Unknown Breed"

print("Loading dog features...")
with open('dog_features.pkl', 'rb') as f:
    dog_features = pickle.load(f)
print("Dog features loaded successfully!")

def find_best_matches(user_features, all_dog_features):
    matches = []
    for dog_path, dog_vec in all_dog_features.items():
        dist = cosine(user_features, dog_vec)
        matches.append((dog_path, dist))
    
    # Sort matches by similarity (lower distance = more similar)
    matches.sort(key=lambda x: x[1])
    return matches  # Returns list of (path, distance) tuples sorted by similarity

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
            
            # Get all matches sorted by similarity
            all_matches = find_best_matches(user_features, dog_features)
            best_dog_image_path = all_matches[0][0]  # Get the first (best) match
            
            # Store all matches in session for later use
            session['current_matches'] = [match[0] for match in all_matches]
            session['current_match_index'] = 0
            
            display_user_path = os.path.join('uploads', filename)
            display_dog_path = best_dog_image_path
            breed_name = get_breed_name(display_dog_path)
            
            return render_template('result.html', 
                                user_image=display_user_path, 
                                result_image=display_dog_path,
                                breed_name=breed_name)

    return render_template('index.html')

@app.route('/Images/<path:filename>')
def serve_dog_image(filename):
    return send_from_directory('Images', filename)

@app.route('/uploads/<path:filename>')
def serve_user_image(filename):
    return send_from_directory('uploads', filename)

@app.route('/next-match', methods=['POST'])
def next_match():
    if 'current_matches' not in session:
        # If no matches in session, find new matches using all dogs
        all_dog_paths = list(dog_features.keys())
        session['current_matches'] = all_dog_paths
        session['current_match_index'] = 0
    
    matches = session['current_matches']
    current_index = session['current_match_index']
    current_breed = get_breed_name(matches[current_index])
    
    # Find all matches of the same breed
    same_breed_matches = [
        (i, path) for i, path in enumerate(matches)
        if get_breed_name(path) == current_breed and i != current_index
    ]
    
    if same_breed_matches:
        # If there are other dogs of the same breed, pick one randomly
        from random import choice
        next_index, next_match_path = choice(same_breed_matches)
    else:
        # If no other dogs of the same breed, move to next match
        next_index = (current_index + 1) % len(matches)
        next_match_path = matches[next_index]
    
    session['current_match_index'] = next_index
    breed_name = get_breed_name(next_match_path)
    
    return render_template('result.html', 
                         user_image=request.form['user_image'],
                         result_image=next_match_path,
                         breed_name=breed_name)

if __name__ == '__main__':
    app.run(debug=True)