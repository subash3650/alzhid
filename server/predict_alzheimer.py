import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

class AlzheimerPredictor:
    """
    Easy-to-use predictor for Alzheimer's detection
    """
    
    def __init__(self, model_path='trained_models/alzheimer_model_final.h5'):
        """
        Load the trained model
        """
        print("=" * 70)
        print("ALZHEIMER'S DISEASE PREDICTOR")
        print("=" * 70)
        
        # Load model
        if not os.path.exists(model_path):
            print(f"\\nERROR: Model not found at {model_path}")
            print("Please train the model first: python train_alzheimer_model.py")
            self.model = None
            return
        
        print(f"\\nLoading model from: {model_path}")
        self.model = load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Load metadata
        metadata_path = 'trained_models/model_metadata.pkl'
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.class_names = metadata['class_names']
            self.stage_map = metadata['stage_map']
            self.input_shape = metadata['input_shape']
        else:
            # Defaults
            self.class_names = ['Mild Impairment', 'Moderate Impairment', 
                               'No Impairment', 'Very Mild Impairment']
            self.stage_map = {
                'No Impairment': 'Stage 0 - Healthy',
                'Very Mild Impairment': 'Stage 1 - Very Mild (MCI)',
                'Mild Impairment': 'Stage 2 - Mild AD',
                'Moderate Impairment': 'Stage 3 - Moderate AD'
            }
            self.input_shape = (176, 208, 3)
        
        # Treatment recommendations
        self.treatments = {
            'No Impairment': [
                'Maintain healthy lifestyle',
                'Regular physical exercise',
                'Cognitive stimulation activities',
                'Balanced diet (Mediterranean diet)',
                'Annual health checkups'
            ],
            'Very Mild Impairment': [
                'Consult neurologist for assessment',
                'Cholinesterase inhibitors (e.g., Donepezil)',
                'Memory training programs',
                'Regular cognitive testing',
                'Consider clinical trial participation'
            ],
            'Mild Impairment': [
                'Neurologist consultation required',
                'Cholinesterase inhibitors (Donepezil, Rivastigmine)',
                'NMDA antagonist (Memantine)',
                'Cognitive behavioral therapy',
                'Caregiver support groups'
            ],
            'Moderate Impairment': [
                'Geriatric specialist consultation',
                'Memantine treatment',
                'Full-time caregiver support',
                'Palliative care planning',
                'Safety measures at home'
            ]
        }
    
    def preprocess_image(self, image_path):
        """
        Prepare image for prediction
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Resize
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        
        # Same preprocessing as training
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        img = cv2.bitwise_and(img, img, mask=mask)
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def predict(self, image_path):
        """
        Analyze an MRI scan and return diagnosis
        """
        if self.model is None:
            print("ERROR: Model not loaded!")
            return None
        
        print(f"\\nAnalyzing: {image_path}")
        
        # Preprocess
        img = self.preprocess_image(image_path)
        img_batch = np.expand_dims(img, axis=0)
        
        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        predicted_class = self.class_names[predicted_idx]
        confidence = predictions[predicted_idx]
        
        # Calculate severity score (0-100%)
        severity_ranges = {
            'No Impairment': (0, 10),
            'Very Mild Impairment': (10, 35),
            'Mild Impairment': (35, 70),
            'Moderate Impairment': (70, 100)
        }
        severity_range = severity_ranges.get(predicted_class, (50, 50))
        severity = int(np.mean(severity_range))
        
        # Compile results
        results = {
            'diagnosis': predicted_class,
            'stage': self.stage_map.get(predicted_class, 'Unknown'),
            'confidence': float(confidence),
            'severity_score': severity,
            'probabilities': {
                name: float(prob)
                for name, prob in zip(self.class_names, predictions)
            },
            'treatments': self.treatments.get(predicted_class, [])
        }
        
        # Print results
        self.print_results(results)
        
        return results
    
    def print_results(self, results):
        """
        Display results in a nice format
        """
        print("\\n" + "=" * 70)
        print("DIAGNOSIS RESULTS")
        print("=" * 70)
        print(f"\\n🧠 Diagnosis: {results['diagnosis']}")
        print(f"📊 Stage: {results['stage']}")
        print(f"✅ Confidence: {results['confidence']*100:.2f}%")
        print(f"⚠️  Severity Score: {results['severity_score']}%")
        
        print("\\n" + "-" * 70)
        print("CLASS PROBABILITIES:")
        for class_name, prob in results['probabilities'].items():
            print(f"  {class_name:25s}: {prob*100:5.2f}%")
        
        print("\\n" + "-" * 70)
        print("RECOMMENDED TREATMENTS:")
        for i, treatment in enumerate(results['treatments'], 1):
            print(f"  {i}. {treatment}")
        
        print("\\n" + "=" * 70)
        print("⚠️  IMPORTANT: This is an AI prediction, not a medical diagnosis.")
        print("Always consult a qualified healthcare professional.")
        print("=" * 70)

# ============================================================
# USAGE EXAMPLE
# ============================================================

def main():
    """
    Example: How to use the predictor
    """
    # Initialize predictor
    predictor = AlzheimerPredictor()
    test_image = "A:/Downloads/images.jpg"
    
    if os.path.exists(test_image):
        results = predictor.predict(test_image)
    else:
        print(f"\\nExample image not found: {test_image}")
        print("\\nTo use this script:")
        print("  predictor = AlzheimerPredictor()")
        print("  results = predictor.predict('path/to/your/mri.jpg')")

if __name__ == "__main__":
    main()
