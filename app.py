from flask import Flask, render_template, request, jsonify, send_from_directory
import base64
import cv2
import numpy as np
import json
import os
from insightface.app import FaceAnalysis
import uuid
from datetime import datetime

app = Flask(__name__, static_folder='static', static_url_path='/static')

# Configure image storage folder
UPLOAD_FOLDER = 'static/facereco/images'
DATA_FILE = 'users_data.json'

# Create directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.route('/api/verify_user', methods=['POST'])
def verify_user():
    try:
        data = request.get_json()
        image_base64 = data.get('image')
        
        if not image_base64:
            return jsonify({'error': 'Missing image'}), 400
        
        # Convert base64 to image
        image_data = base64.b64decode(image_base64.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
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
        
        if best_similarity >= 0.8:
            return jsonify({
                'success': True,
                'match': True,
                'user': best_match,
                'similarity': best_similarity
            })
        else:
            return jsonify({
                'success': True,
                'match': False,
                'similarity': best_similarity
            })
            
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

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
    app.run(debug=True)
