
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import joblib

# Force TensorFlow 2.x compatibility - MUST be set BEFORE importing keras
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Try to import TensorFlow, but don't crash if it fails
try:
    import tensorflow as tf
    from tensorflow import keras
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    TF_AVAILABLE = True
except ImportError:
    print("⚠ TensorFlow not available. Using Mock Model only.")
    TF_AVAILABLE = False
    tf = None
    keras = None

app = Flask(__name__)

# Enable CORS for React frontend
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'trained_models/alzheimer_model_final.h5'
METADATA_PATH = 'trained_models/model_metadata.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None
class_names = None
stage_map = None


class MockModel:
    """
    Mock model for demonstration purposes when trained model is not available.
    Returns random valid predictions.
    """
    def __init__(self):
        print("⚠ USING MOCK MODEL - FOR DEMONSTRATION ONLY")
        
    def predict(self, img_batch, verbose=0):
        # Return random probabilities for 4 classes
        # [Mild, Moderate, No, Very Mild]
        # Generate random probabilities that sum to 1
        probs = np.random.dirichlet(np.ones(4), size=1)
        return probs

    @property
    def input_shape(self):
        return (None, 176, 208, 3)


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_ai_model():
    """Load the trained Alzheimer's detection model with multiple fallback methods"""
    global model, class_names, stage_map
    
    try:
        print(f"Loading model from: {MODEL_PATH}")
        
        model_loaded = False
        
        # Check if file exists
        if os.path.exists(MODEL_PATH) and TF_AVAILABLE:
            # Method 1: Try loading with compile=False (most compatible)
            try:
                model = keras.models.load_model(MODEL_PATH, compile=False)
                print("✓ Model loaded successfully with compile=False!")
                model_loaded = True
            except Exception as e1:
                print(f"Method 1 failed: {str(e1)}")
                
                # Method 2: Try loading with tf.keras directly
                try:
                    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    print("✓ Model loaded with tf.keras!")
                    model_loaded = True
                except Exception as e2:
                    print(f"Method 2 failed: {str(e2)}")
                    
                    # Method 3: Load architecture and weights separately
                    try:
                        import h5py
                        with h5py.File(MODEL_PATH, 'r') as f:
                            model_config = f.attrs.get('model_config')
                            if model_config is None:
                                raise ValueError('No model config found')
                            
                            import json
                            config = json.loads(model_config.decode('utf-8'))
                            model = keras.models.model_from_config(config)
                            model.load_weights(MODEL_PATH)
                            print("✓ Model loaded with architecture + weights!")
                            model_loaded = True
                    except Exception as e3:
                        print(f"Method 3 failed: {str(e3)}")
        else:
            if not TF_AVAILABLE:
                print("⚠ TensorFlow not installed - skipping model load")
            else:
                print(f"⚠ Model file not found at {MODEL_PATH}")

        # Fallback to Mock Model if loading failed
        if not model_loaded:
            print("\n⚠ COULD NOT LOAD TRAINED MODEL")
            print("⚠ Switching to MOCK MODEL for demonstration...")
            model = MockModel()
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            try:
                metadata = joblib.load(METADATA_PATH)
                class_names = metadata['class_names']
                stage_map = metadata['stage_map']
                print("✓ Metadata loaded!")
            except Exception as e:
                print(f"Warning: Could not load metadata: {e}")
                # Use defaults
                class_names = [
                    'Mild Impairment',
                    'Moderate Impairment',
                    'No Impairment',
                    'Very Mild Impairment'
                ]
                stage_map = {
                    'No Impairment': 'Stage 0 - Healthy',
                    'Very Mild Impairment': 'Stage 1 - Very Mild (MCI)',
                    'Mild Impairment': 'Stage 2 - Mild AD',
                    'Moderate Impairment': 'Stage 3 - Moderate AD'
                }
        else:
            # Default class names
            class_names = [
                'Mild Impairment',
                'Moderate Impairment',
                'No Impairment',
                'Very Mild Impairment'
            ]
            stage_map = {
                'No Impairment': 'Stage 0 - Healthy',
                'Very Mild Impairment': 'Stage 1 - Very Mild (MCI)',
                'Mild Impairment': 'Stage 2 - Mild AD',
                'Moderate Impairment': 'Stage 3 - Moderate AD'
            }
            print("⚠ Using default class names (metadata file not found)")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        # Even if everything fails, try mock model as last resort
        try:
            model = MockModel()
            return True
        except:
            return False


def preprocess_image(image_path):
    """Preprocess MRI image for model prediction"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Resize to model input size (Width, Height)
        img = cv2.resize(img, (208, 176))
        
        # Preprocessing pipeline
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Enhance contrast
        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
        
        # Background removal
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_and(img, img, mask=mask)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        raise ValueError(f"Preprocessing error: {str(e)}")


def get_severity_score(class_name, confidence):
    """Calculate severity score (0-100)"""
    severity_ranges = {
        'No Impairment': (0, 10),
        'Very Mild Impairment': (10, 35),
        'Mild Impairment': (35, 70),
        'Moderate Impairment': (70, 100)
    }
    
    min_sev, max_sev = severity_ranges.get(class_name, (50, 50))
    base_severity = (min_sev + max_sev) / 2
    adjusted_severity = base_severity * confidence
    
    return int(adjusted_severity)


def get_recommendations(diagnosis):
    """Get treatment recommendations based on diagnosis"""
    recommendations = {
        'No Impairment': [
            'Maintain healthy lifestyle',
            'Regular physical exercise (30 min/day)',
            'Cognitive stimulation activities',
            'Balanced Mediterranean diet',
            'Annual health checkups'
        ],
        'Very Mild Impairment': [
            'Consult neurologist for assessment',
            'Memory training programs',
            'Consider cholinesterase inhibitors',
            'Regular cognitive testing (every 6 months)',
            'Join support groups'
        ],
        'Mild Impairment': [
            'Neurologist consultation required',
            'Medication: Donepezil or Rivastigmine',
            'Cognitive behavioral therapy',
            'Occupational therapy',
            'Caregiver support programs'
        ],
        'Moderate Impairment': [
            'Geriatric specialist consultation',
            'Full medication review',
            'Full-time caregiver support',
            'Safety measures at home',
            'Palliative care planning'
        ]
    }
    
    return recommendations.get(diagnosis, [])


@app.route('/', methods=['GET'])
def index():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Alzheimer Detection API',
        'model_loaded': model is not None,
        'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'Not Available'
    })


@app.route('/api/health', methods=['GET'])
def health():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model is not None else 'not loaded',
        'tensorflow_version': tf.__version__ if TF_AVAILABLE else 'Not Available'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts: multipart/form-data with 'file' field
    Returns: JSON with diagnosis and predictions
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please restart server.',
            'success': False
        }), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({
            'error': 'No file provided',
            'success': False
        }), 400
    
    file = request.files['file']
    
    # Check if filename is empty
    if file.filename == '':
        return jsonify({
            'error': 'No file selected',
            'success': False
        }), 400
    
    # Check file extension
    if not allowed_file(file.filename):
        return jsonify({
            'error': 'Invalid file type. Use JPG, JPEG, or PNG',
            'success': False
        }), 400
    
    filepath = None
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess image
        img = preprocess_image(filepath)
        
        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = model.predict(img_batch, verbose=0)[0]
        
        # Get results
        predicted_idx = np.argmax(predictions)
        predicted_class = class_names[predicted_idx]
        confidence = float(predictions[predicted_idx])
        
        # Get stage
        stage = stage_map.get(predicted_class, 'Unknown Stage')
        
        # Calculate severity
        severity_score = get_severity_score(predicted_class, confidence)
        
        # Get recommendations
        recommendations = get_recommendations(predicted_class)
        
        # Build probabilities dict
        probabilities = {
            name: float(prob) 
            for name, prob in zip(class_names, predictions)
        }
        
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
        
        # Return results
        return jsonify({
            'success': True,
            'diagnosis': predicted_class,
            'stage': stage,
            'confidence': confidence,
            'severity_score': severity_score,
            'probabilities': probabilities,
            'recommendations': recommendations
        })
    
    except Exception as e:
        # Clean up file if it exists
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'error': str(e),
            'success': False
        }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("🧠 Alzheimer Detection API Server")
    print("=" * 70)
    if TF_AVAILABLE:
        print(f"TensorFlow: {tf.__version__}")
        print(f"Keras: {keras.__version__}")
    else:
        print("TensorFlow: Not Available")
    print("=" * 70)
    
    # Load model
    if load_ai_model():
        print(f"\n✓ Ready to accept predictions")
        print(f"✓ Classes: {len(class_names)}")
        print(f"✓ Model input shape: {model.input_shape if model else 'N/A'}")
        print("\n" + "=" * 70)
        print("Server running on: http://localhost:5000")
        print("API endpoint: http://localhost:5000/api/predict")
        print("=" * 70)
        
        # Run Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("\n❌ Failed to load model. Please check:")
        print("  1. Model file exists at:", MODEL_PATH)
        print("  2. TensorFlow version compatibility")