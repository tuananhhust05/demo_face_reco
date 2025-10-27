# Face Recognition App

Simple face recognition application built with Flask and InsightFace.

## Features

- **Create New User** (`/facereco/create`): Take face photo and save user information
- **Face Verification** (`/facereco/verify`): Search users by face

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

Access the application at: http://localhost:5000

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
    └── images/           # User images storage
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

## Notes

- Application uses cosine similarity threshold ≥ 0.8 to determine matches
- Images are saved in `static/images/` directory and are publicly accessible
- User data is stored in JSON file `users_data.json`

## System Requirements

- Python 3.8+
- Webcam
- Internet connection (to download model on first run)
