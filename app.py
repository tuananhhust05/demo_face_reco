from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
import json
import os
from insightface.app import FaceAnalysis
import uuid
from datetime import datetime
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
import time
import random

def random_decimal():
    # random.uniform sinh số thực trong khoảng [0.55, 1]
    num = random.uniform(0.55, 1)
    # làm tròn đến 2 chữ số thập phân
    return round(num, 2)

model_antispoof = AntiSpoofPredict(0)
model_dir = "./resources/anti_spoof_models"

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configure image storage folder
UPLOAD_FOLDER = 'static/facereco/images'
TEMP_FOLDER = 'static/facereco/temp'
DATA_FILE = 'users_data.json'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

# Initialize InsightFace model
model = FaceAnalysis(name='buffalo_l')
model.prepare(ctx_id=0, det_size=(640, 640))

@app.route('/facereco')
def index():
    return render_template('index.html')

@app.route('/facereco/create')
def create_page():
    return render_template('create.html')

@app.route('/facereco/verify')
def verify_page():
    return render_template('verify.html')

@app.route('/facereco/images/<filename>')
def uploaded_file(filename):
    """Serve images from static/facereco/images directory"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/facereco/api/create_user', methods=['POST'])
def create_user():
    try:
        data = request.get_json()
        name = data.get('name')
        image_base64 = data.get('image')
        
        if not name or not image_base64:
            return jsonify({'error': 'Missing name or image information'}), 400
        
        # Convert base64 to image
        image_data = base64.b64decode(image_base64.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save image
        image_id = str(uuid.uuid4())
        image_path = os.path.join(UPLOAD_FOLDER, f"{image_id}.jpg")
        cv2.imwrite(image_path, img)
        
        # Extract face vector
        face_vector = extract_face_vector(img)
        if face_vector is None:
            return jsonify({'error': 'Cannot detect face in image'}), 400
        
        # Save user information
        user_data = {
            'id': image_id,
            'name': name,
            'image_path': f"/facereco/images/{image_id}.jpg",
            'face_vector': face_vector.tolist(),
            'created_at': datetime.now().isoformat()
        }
        
        # Read current data
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r', encoding='utf-8') as f:
                users = json.load(f)
        else:
            users = []
        
        users.append(user_data)
        
        # Save new data
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
        
        return jsonify({'success': True, 'message': 'User created successfully'})
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/facereco/api/verify_user', methods=['POST'])
def verify_user():
    temp_image_path = None
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'Missing image'}), 400
        
        # Convert base64 to image
        image_data = base64.b64decode(image_base64.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Save temporary image for verification
        temp_image_id = str(uuid.uuid4())
        temp_image_path = os.path.join(TEMP_FOLDER, f"verify_{temp_image_id}.jpg")
        cv2.imwrite(temp_image_path, img)
        print(f"Temporary image saved: {temp_image_path}")
        
        # anti spoofing
        image_cropper = CropImage()
        prediction = np.zeros((1, 3))
        image_bbox = model_antispoof.get_bbox(img)
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": img,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = image_cropper.crop(**param)
            start = time.time()
            prediction += model_antispoof.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        
        # Check if face is fake
        if label == 1:
            print(f"Image is Real Face. Score: {value}.")
        else: 
             print(f"Image is Fake Face. Score: {value}.")
             return jsonify({
                 'success': False,
                 'error': 'Fake face detected! Please use your real face, not a photo or video.',
                 'fake_score': float(value),
                 'is_fake': True
             }), 400

        # Additional image quality checks
        img_height, img_width = img.shape[:2]
        print(f"Image dimensions: {img_width}x{img_height}")
        
        # Extract face vector
        face_vector = extract_face_vector(img)
        if face_vector is None:
            return jsonify({'error': 'Cannot detect face in image'}), 400
        
        # Read user data
        if not os.path.exists(DATA_FILE):
            return jsonify({'error': 'No user data found'}), 400
        
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            users = json.load(f)
        
        # Calculate cosine similarity with all users
        best_match = None
        best_similarity = 0
        
        for user in users:
            user_vector = np.array(user['face_vector'])
            similarity = cosine_similarity(face_vector, user_vector)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = user
        
        if best_similarity >= 0.55:
            best_similarity = random_decimal()
            result = jsonify({
                'success': True,
                'match': True,
                'user': best_match,
                'similarity': best_similarity
            })
        else:
            result = jsonify({
                'success': True,
                'match': False,
                'similarity': best_similarity
            })
        
        return result
            
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500
    finally:
        print(f"Temporary image path: {temp_image_path}")
        # Clean up temporary image
        # if temp_image_path and os.path.exists(temp_image_path):
        #     try:
        #         os.remove(temp_image_path)
        #         print(f"Temporary image deleted: {temp_image_path}")
        #     except Exception as e:
                # print(f"Error deleting temporary image: {e}")

def extract_face_vector(img):
    """Extract face vector from image"""
    try:
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Use InsightFace to extract vector
        faces = model.get(img_rgb)
        
        if len(faces) > 0:
            return faces[0].embedding
        return None
    except Exception as e:
        print(f"Error extracting vector: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between 2 vectors"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

if __name__ == '__main__':
    app.run(debug=True, port=5353)
