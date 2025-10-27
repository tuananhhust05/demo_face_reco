# Face Recognition App

Simple face recognition application built with Flask and InsightFace.

## Features

- **Create New User** (`/facereco/create`): Take face photo and save user information
- **Face Verification** (`/facereco/verify`): Search users by face
- **Real-time Face Detection**: Live camera integration with face detection

## Installation

1. Install Python 3.8+ if not already installed

2. Create virtual environment (recommended):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install required libraries:
```bash
pip install -r requirements.txt
```

4. Download InsightFace model (will be downloaded automatically on first run):
```bash
python -c "from insightface.model_zoo import get_model; get_model('buffalo_l')"
```

## Running the Application

```bash
python app.py
```

Access the application at: http://localhost:5353

## Project Structure

```
demoface_reco/
├── app.py                 # Main Flask application file
├── requirements.txt       # Required libraries list
├── users_data.json       # User data storage file (auto-created)
├── templates/            # HTML templates directory
│   ├── index.html        # Home page
│   ├── create.html       # Create user page
│   └── verify.html       # Verification page
└── static/               # Static files directory
    └── facereco/         # Face recognition files
        ├── images/       # User images storage
        └── temp/         # Temporary verification images (auto-deleted)
```

## Usage

### Create New User
1. Go to `/facereco/create`
2. Enter user name
3. Click "Start Camera" to enable webcam
4. Click "Capture Photo" to take face photo
5. Click "Create User" to save information

### Face Verification
1. Go to `/facereco/verify`
2. Click "Start Camera" to enable webcam
3. Click "Capture Photo" to take face photo for verification
4. Click "Verify" to search for users

## Configuration

### Adjusting Face Similarity Threshold

You can easily adjust the face matching threshold in `config.py`:

```python
FACE_SIMILARITY_THRESHOLD = 0.55  # Adjust between 0.55 and 1.0
```

**Threshold Guidelines:**
- **0.55-0.65**: Very lenient (more matches, more false positives)
- **0.65-0.75**: Moderate (balanced)
- **0.75-0.85**: Strict (fewer matches, more accurate)
- **0.85-1.0**: Very strict (high accuracy, may miss some matches)

### Quick Threshold Adjustment

Use the provided script:
```bash
python adjust_threshold.py
```

## Notes

- Application uses configurable cosine similarity threshold (default: 0.55)
- Images are saved in `static/facereco/images/` directory and are publicly accessible
- User data is stored in JSON file `users_data.json`
- **Temporary verification images** are automatically saved to `static/facereco/temp/` and deleted after verification
- **Anti-spoofing protection** prevents fake face attacks

## System Requirements

- Python 3.8+
- Webcam
- Internet connection (to download model on first run)
