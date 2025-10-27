# Face Recognition Configuration
# Adjust these values to fine-tune the system

# Face similarity threshold (0.55 to 1.0)
# Lower values = more lenient matching (more false positives)
# Higher values = more strict matching (more false negatives)
FACE_SIMILARITY_THRESHOLD = 0.55

# Anti-spoofing settings
ANTI_SPOOF_ENABLED = True

# Image quality settings
MAX_IMAGE_SIZE = (1920, 1080)
IMAGE_QUALITY = 95

# Server settings
DEBUG_MODE = True
PORT = 5353
HOST = '0.0.0.0'

# File paths
UPLOAD_FOLDER = 'static/facereco/images'
TEMP_FOLDER = 'static/facereco/temp'
DATA_FILE = 'users_data.json'

# Model settings
FACE_MODEL_NAME = 'buffalo_l'
