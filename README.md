# 🐶 Dog Look-Alike Finder 🧑

Ever wondered which dog breed you resemble? This fun web application uses a deep learning model to analyze a photo of your face and find your canine twin from a dataset of over 100 dog breeds!


This project is a hands-on introduction to building a full-stack machine learning application, covering everything from data processing and feature extraction to building a web interface with Flask.

## ✨ Features

-   **Simple Web Interface:** Easy-to-use webpage for uploading your photo.
-   **AI-Powered Face Detection:** Uses OpenCV to automatically detect and crop the human face from the uploaded image, ensuring more accurate comparisons.
-   **Deep Learning Feature Extraction:** Employs a powerful pre-trained Convolutional Neural Network (VGG16) to understand the unique visual features of a face.
-   **Efficient Similarity Search:** Compares facial features using Cosine Similarity to find the best match quickly.

---

## 🔧 How It Works

The application's logic is broken down into a simple but powerful machine learning pipeline:

1.  **Data Pre-processing (One-time step):** The `process_dataset.py` script first runs through the entire Stanford Dogs Dataset. For each dog image, it performs a center-crop and uses the VGG16 model to extract a **feature vector** (a numerical representation of the image). All these vectors are saved to a single file, `dog_features.pkl`, for fast lookups later.

2.  **User Upload:** A user uploads their photo via the Flask web application.

3.  **Human Face Detection:** The backend uses an OpenCV Haar Cascade classifier to find the user's face in the uploaded photo and crops the image to that region.

4.  **Feature Extraction:** The cropped face image is then passed to the same VGG16 model to generate its unique feature vector.

5.  **Comparison:** The application calculates the **Cosine Similarity** between the user's feature vector and every dog vector loaded from `dog_features.pkl`. The dog with the highest similarity score (lowest cosine distance) is the winner!

6.  **Display Results:** The original photo and the winning dog photo are displayed to the user.

---

## 🛠️ Technology Stack

-   **Backend:** Python, Flask
-   **Machine Learning:** TensorFlow, Keras
-   **Computer Vision:** OpenCV, Pillow
-   **Scientific Computing:** NumPy, SciPy
-   **Frontend:** HTML, CSS

---

## 🚀 Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

-   Python 3.8+
-   `pip` (Python package installer)

### 1. Clone the Repository

```bash
git clone https://github.com/jasperlawrencee/watda-dog-r-u.git
cd watda-dog-r-u
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to keep project dependencies isolated.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Download Required Data

-   **Stanford Dogs Dataset:**
    -   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset?select=images).
    -   Unzip the file and place the `images/Images` folder into the root of the project directory. Make sure the folder is named **`Images`**.

-   **Haar Cascade Classifier:**
    -   Download the `haarcascade_frontalface_default.xml` file from [OpenCV's GitHub repository](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml).
    -   Place this file in the root of the project directory.

### 5. Pre-process the Dog Dataset

This is a crucial one-time step. Run the processing script to generate the feature vectors for the entire dog dataset. This will create the `dog_features.pkl` file.

**Note:** This process can take 15-30 minutes depending on your computer's performance.

```bash
python process_dataset.py
```

---

## ▶️ How to Run the Application

Once the setup is complete, you can start the Flask web server.

```bash
flask --app app run
```

Open your web browser and navigate to:
**http://127.0.0.1:5000**

You should now see the application's homepage. Upload a photo and enjoy!

---

## 📁 Project Structure

```
.
├── app.py                                  # Main Flask application
├── feature_extractor.py                    # Core ML/CV functions (cropping, feature extraction)
├── process_dataset.py                      # Script to pre-process the dog dataset
├── requirements.txt                        # Project dependencies
├── haarcascade_frontalface_default.xml     # OpenCV face detector
├── dog_features.pkl                        # Generated file of dog feature vectors
├── Images/                                 # Folder for the Stanford Dogs Dataset
│   └── n02085620-Chihuahua/
│   └── ...
├── templates/                              # Flask HTML templates
│   ├── index.html
│   └── result.html
└── uploads/                                # Stores user-uploaded images temporarily
```

---

## 🙏 Acknowledgements

-   This project uses the **Stanford Dogs Dataset**.
-   Feature extraction is powered by the **VGG16** model, pre-trained on ImageNet.
-   Face detection is performed using **OpenCV**.
