import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import joblib

# Disable warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("ALZHEIMER'S DISEASE DETECTION - AI TRAINING")
print("=" * 70)
print("\\nThis will train an AI model to detect Alzheimer's from MRI scans")
print("Expected time: 2-3 hours with GPU, 8-12 hours without GPU")
print("=" * 70)

# ============================================================
# CONFIGURATION - ADJUST THESE IF NEEDED
# ============================================================

class Config:
    # YOUR dataset path (from the image you showed)
    DATASET_PATH = "alzheimer_dataset/Combined Dataset"
    
    # Where to save the trained models
    MODEL_SAVE_PATH = "trained_models"
    CHECKPOINT_PATH = "model_checkpoints"
    
    # Image size for the AI
    IMG_HEIGHT = 176
    IMG_WIDTH = 208
    IMG_CHANNELS = 3
    
    # Training settings (you can change these)
    BATCH_SIZE = 8          # How many images to process at once
    EPOCHS = 50              # How many times to go through all data
    LEARNING_RATE = 0.0001   # How fast the AI learns
    
    # YOUR class names (from your dataset folders)
    CLASS_NAMES = [
        'Mild Impairment',
        'Moderate Impairment',
        'No Impairment',
        'Very Mild Impairment'
    ]
    
    # Clinical stage mapping
    STAGE_MAP = {
        'No Impairment': 'Stage 0 - Healthy',
        'Very Mild Impairment': 'Stage 1 - Very Mild (MCI)',
        'Mild Impairment': 'Stage 2 - Mild AD',
        'Moderate Impairment': 'Stage 3 - Moderate AD'
    }

# Create folders for saving
os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(Config.CHECKPOINT_PATH, exist_ok=True)

# ============================================================
# STEP 1: LOAD AND PREPROCESS IMAGES
# ============================================================

def preprocess_image(img_path):
    """
    Prepare an MRI image for the AI model
    """
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    # Resize to standard size
    img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast
    gray = cv2.equalizeHist(gray)
    
    # Remove noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Remove skull (keep only brain)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    # Normalize pixel values to 0-1
    img = img.astype(np.float32) / 255.0
    
    return img

def load_dataset(dataset_path, subset='train'):
    """
    Load all images from your dataset
    """
    subset_path = os.path.join(dataset_path, subset)
    images = []
    labels = []
    
    print(f"\\n{'=' * 70}")
    print(f"Loading {subset.upper()} images...")
    print(f"{'=' * 70}")
    
    if not os.path.exists(subset_path):
        print(f"ERROR: Folder not found: {subset_path}")
        print("Please check your dataset path!")
        return None, None
    
    # Load images from each class folder
    for class_idx, class_name in enumerate(Config.CLASS_NAMES):
        class_path = os.path.join(subset_path, class_name)
        
        if not os.path.exists(class_path):
            print(f"WARNING: Class folder not found: {class_name}")
            continue
        
        # Find all image files
        image_files = list(Path(class_path).glob('*.jpg')) + \
                      list(Path(class_path).glob('*.png'))
        
        print(f"\\nLoading: {class_name}")
        print(f"  Found {len(image_files)} images")
        
        # Process each image
        loaded = 0
        for img_path in image_files:
            img = preprocess_image(str(img_path))
            if img is not None:
                images.append(img)
                labels.append(class_idx)
                loaded += 1
            
            # Show progress
            if loaded % 100 == 0:
                print(f"  Processed: {loaded}/{len(image_files)}", end='\\r')
        
        print(f"  ✓ Loaded: {loaded} images")
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"\\n{'=' * 70}")
    print(f"Total {subset} images loaded: {len(images)}")
    print(f"{'=' * 70}")
    
    return images, labels

# ============================================================
# STEP 2: BUILD THE AI MODEL
# ============================================================

def build_model():
    """
    Create the AI model architecture
    Uses VGG16 (a proven image recognition AI)
    """
    print("\\n" + "=" * 70)
    print("BUILDING AI MODEL")
    print("=" * 70)
    
    # Load pre-trained VGG16 (already knows about images)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=(Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)
    )
    
    # Freeze early layers (keep their knowledge)
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    
    # Build complete model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(len(Config.CLASS_NAMES), activation='softmax')
    ])
    
    print("\\nModel Architecture:")
    model.summary()
    
    return model

# ============================================================
# STEP 3: TRAIN THE MODEL
# ============================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the AI model on your data
    """
    print("\\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print("=" * 70)
    
    # Compile model (prepare for training)
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Calculate class weights (handle imbalanced data)
    unique, counts = np.unique(y_train, return_counts=True)
    class_weights = {
        i: len(y_train) / (len(unique) * count)
        for i, count in zip(unique, counts)
    }
    
    print("\\nClass weights:", class_weights)
    
    # Set up callbacks
    checkpoint = ModelCheckpoint(
        os.path.join(Config.CHECKPOINT_PATH, 'best_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    
    # Data augmentation (create variations of images)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.15,
        fill_mode='nearest'
    )
    
    print("\\nTraining started...")
    print("This will take several hours. Go grab coffee! ☕")
    print("=" * 70)
    
    # Train!
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=Config.BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=Config.EPOCHS,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(Config.MODEL_SAVE_PATH, 'alzheimer_model_final.h5'))
    print(f"\\n✓ Model saved: {Config.MODEL_SAVE_PATH}/alzheimer_model_final.h5")
    
    return model, history

# ============================================================
# STEP 4: EVALUATE THE MODEL
# ============================================================

def evaluate_model(model, X_test, y_test):
    """
    Test how accurate the model is
    """
    print("\\n" + "=" * 70)
    print("EVALUATING MODEL PERFORMANCE")
    print("=" * 70)
    
    # Make predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\\nOverall Accuracy: {accuracy*100:.2f}%")
    
    # Detailed report
    print("\\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=Config.CLASS_NAMES))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=Config.CLASS_NAMES,
                yticklabels=Config.CLASS_NAMES)
    plt.title('Confusion Matrix - Model Performance')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'confusion_matrix.png'))
    print(f"\\n✓ Confusion matrix saved: {Config.MODEL_SAVE_PATH}/confusion_matrix.png")
    plt.close()
    
    return accuracy

def plot_training_history(history):
    """
    Show how the model learned over time
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss Over Time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.MODEL_SAVE_PATH, 'training_history.png'))
    print(f"✓ Training history saved: {Config.MODEL_SAVE_PATH}/training_history.png")
    plt.close()

# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Run the complete training process
    """
    print("\\n\\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "ALZHEIMER'S AI TRAINING START" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # Check GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"\\n✓ GPU detected: {len(gpus)} device(s)")
        print("Training will be FAST! 🚀")
    else:
        print("\\n⚠ No GPU detected - training will be SLOW (8-12 hours)")
        print("Consider using Google Colab for free GPU:")
        print("https://colab.research.google.com/")
    
    # Load training data
    X_train, y_train = load_dataset(Config.DATASET_PATH, 'train')
    X_test, y_test = load_dataset(Config.DATASET_PATH, 'test')
    
    if X_train is None or X_test is None:
        print("\\n✗ ERROR: Could not load dataset!")
        print("Please check:")
        print("  1. Dataset path is correct")
        print("  2. Folder structure matches your dataset")
        print("  3. Images are in .jpg or .png format")
        return
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        random_state=42,
        stratify=y_train
    )
    
    print(f"\\n{'=' * 70}")
    print("DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"Training:   {len(X_train):5d} images")
    print(f"Validation: {len(X_val):5d} images")
    print(f"Test:       {len(X_test):5d} images")
    print(f"Total:      {len(X_train) + len(X_val) + len(X_test):5d} images")
    print(f"{'=' * 70}")
    
    # Build model
    model = build_model()
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    
    # Plot training history
    plot_training_history(history)
    
    # Save metadata
    metadata = {
        'class_names': Config.CLASS_NAMES,
        'stage_map': Config.STAGE_MAP,
        'accuracy': float(accuracy),
        'input_shape': (Config.IMG_HEIGHT, Config.IMG_WIDTH, Config.IMG_CHANNELS)
    }
    joblib.dump(metadata, os.path.join(Config.MODEL_SAVE_PATH, 'model_metadata.pkl'))
    
    # Final summary
    print("\\n\\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "TRAINING COMPLETE! 🎉" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print(f"\\nFinal Test Accuracy: {accuracy*100:.2f}%")
    print(f"\\nSaved files in '{Config.MODEL_SAVE_PATH}/' folder:")
    print("  ✓ alzheimer_model_final.h5      - Trained AI model")
    print("  ✓ model_metadata.pkl            - Model information")
    print("  ✓ confusion_matrix.png          - Performance visualization")
    print("  ✓ training_history.png          - Training progress")
    print("\\nNext step: Use this model to make predictions!")
    print("Run: python predict_alzheimer.py")

if __name__ == "__main__":
    main()
