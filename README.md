# 🧠 AI-Based Alzheimer's Disease Detection System

**A complete deep learning system for early detection of Alzheimer's Disease using MRI brain scans with React frontend and Flask backend.**

---

## 📋 Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Step-by-Step Installation](#step-by-step-installation)
- [Running the Application](#running-the-application)
- [Training Your Own Model](#training-your-own-model)
- [API Documentation](#api-documentation)
- [Troubleshooting](#troubleshooting)

---

## 🌟 Features

- ✅ **Deep Learning Classification**: VGG16-based CNN for 4-stage Alzheimer's detection
- ✅ **Modern Web Interface**: Beautiful React UI with real-time analysis
- ✅ **Treatment Recommendations**: AI-powered therapy and medication suggestions
- ✅ **Severity Assessment**: Visual indicators and confidence scores
- ✅ **Mock Model Support**: Works without trained models for demonstration

---

## 📁 Project Structure

```
alzhid/
│
├── README.md              # This file
├── client/                # React Frontend (Port 3000)
│   ├── src/
│   │   ├── App.jsx       # Main React component
│   │   ├── App.css       # Styling
│   │   ├── index.css     # Global styles
│   │   └── main.jsx      # Entry point
│   ├── package.json      # Frontend dependencies
│   ├── vite.config.js    # Vite configuration
│   └── index.html        # HTML template
│
└── server/               # Flask Backend (Port 5000)
    ├── app.py            # Main Flask API server ⭐
    ├── train_alzheimer_model.py  # Model training script
    ├── requirements.txt  # Python dependencies
    ├── alzheimer_dataset/  # Place your dataset here
    ├── trained_models/   # Trained models saved here
    └── uploads/          # Temporary image uploads
```

---

## ✅ Prerequisites

### Required Software

| Software | Version | Download Link |
|----------|---------|---------------|
| **Python** | 3.9 or higher | [python.org](https://www.python.org/downloads/) |
| **Node.js** | 16.x or higher | [nodejs.org](https://nodejs.org/) |
| **npm** | Comes with Node.js | - |
| **pip** | Comes with Python | - |

### Check Your Installation

```bash
# Check Python version
python --version
# Should show: Python 3.9.x or higher

# Check Node.js version
node --version
# Should show: v16.x.x or higher

# Check npm version
npm --version
# Should show: 8.x.x or higher
```

---

## 🚀 Step-by-Step Installation

### STEP 1: Clone or Download the Project

If you have Git:
```bash
git clone <your-repo-url>
cd alzhid
```

Or simply navigate to your project folder:
```bash
cd a:/OneDrive/Desktop/alzhid
```

---

### STEP 2: Set Up Backend (Python/Flask)

#### 2.1 Navigate to Server Folder
```bash
cd server
```

#### 2.2 (Optional) Create Virtual Environment
**Recommended to avoid dependency conflicts**

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` in your terminal when activated.

#### 2.3 Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies being installed:**
- `flask==3.0.0` - Web framework
- `flask-cors==4.0.0` - CORS support
- `werkzeug==3.0.1` - WSGI utilities
- `opencv-python==4.9.0.80` - Image processing
- `numpy==1.26.4` - Numerical operations
- `joblib==1.3.2` - Model serialization

**Optional (for training models):**
- `tensorflow==2.15.0` - Deep learning
- `keras==2.15.0` - Neural networks
- `matplotlib==3.8.3` - Plotting
- `seaborn==0.13.2` - Visualization
- `scikit-learn==1.4.1.post1` - ML utilities

> **Note**: If TensorFlow installation fails, the server will still work with Mock Model mode.

#### 2.4 Verify Installation
```bash
python -c "import flask; print('Flask:', flask.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
```

---

### STEP 3: Set Up Frontend (React)

#### 3.1 Navigate to Client Folder
```bash
cd ../client
# or from root: cd client
```

#### 3.2 Install Node Dependencies
```bash
npm install
```

**Dependencies being installed:**
- `react@19.2.0` - UI framework
- `react-dom@19.2.0` - React DOM renderer
- `axios@1.13.2` - HTTP client
- `vite@7.2.2` - Build tool

**Installation time:** ~30-60 seconds

#### 3.3 Verify Installation
```bash
npm list --depth=0
```

Should show all packages installed successfully.

---

## ▶️ Running the Application

### METHOD 1: Run Both Servers Manually

#### Terminal 1 - Start Backend
```bash
cd server
python app.py
```

**Expected Output:**
```
⚠ TensorFlow not available. Using Mock Model only.
======================================================================
🧠 Alzheimer Detection API Server
======================================================================
⚠ USING MOCK MODEL - FOR DEMONSTRATION ONLY
✓ Ready to accept predictions
✓ Classes: 4
======================================================================
Server running on: http://localhost:5000
API endpoint: http://localhost:5000/api/predict
======================================================================
 * Running on http://0.0.0.0:5000
 * Debugger is active!
```

**Backend is ready when you see:** `Running on http://0.0.0.0:5000`

#### Terminal 2 - Start Frontend
```bash
cd client
npm run dev
```

**Expected Output:**
```
  VITE v7.2.2  ready in 589 ms

  ➜  Local:   http://localhost:3000/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

**Frontend is ready when you see:** `Local: http://localhost:3000/`

---

### METHOD 2: Run in Background (Windows)

**Start Backend:**
```bash
cd server
start python app.py
```

**Start Frontend:**
```bash
cd client
start npm run dev
```

---

## 🎯 Using the Application

### STEP 1: Open Your Browser
Navigate to: **http://localhost:3000**

### STEP 2: Upload MRI Image
1. Click **"Click to select MRI image"** or drag & drop
2. Select a brain MRI scan (`.jpg`, `.jpeg`, or `.png`)
3. Preview will appear

### STEP 3: Analyze
1. Click **"Analyze Image"** button
2. Wait 2-3 seconds for processing

### STEP 4: View Results
You'll see:
- **Diagnosis**: Classification result (No Impairment, Very Mild, Mild, Moderate)
- **Stage**: Clinical stage information
- **Confidence Score**: Model confidence (0-100%)
- **Severity Score**: Risk level (0-100%)
- **Class Probabilities**: Breakdown of all predictions
- **Recommendations**: Treatment and lifestyle suggestions

### STEP 5: Analyze Another Image
Click **"Analyze Another Image"** to reset and try a new scan.

---

## 🎓 Training Your Own Model (Optional)

### Prerequisites for Training
- TensorFlow installed: `pip install tensorflow`
- MRI dataset in correct format
- GPU recommended (training takes 2-3 hours with GPU, 8-12 hours with CPU)

### STEP 1: Prepare Your Dataset

Create this folder structure:
```
server/alzheimer_dataset/Combined Dataset/
├── train/
│   ├── No Impairment/          # Healthy brain scans
│   ├── Very Mild Impairment/   # MCI scans
│   ├── Mild Impairment/        # Mild AD scans
│   └── Moderate Impairment/    # Moderate AD scans
└── test/
    ├── No Impairment/
    ├── Very Mild Impairment/
    ├── Mild Impairment/
    └── Moderate Impairment/
```

**Image Requirements:**
- Format: `.jpg` or `.png`
- Content: Brain MRI scans
- Recommended: 100+ images per class

### STEP 2: Start Training

```bash
cd server
python train_alzheimer_model.py
```

**What happens during training:**
1. Loads and preprocesses images (skull stripping, normalization)
2. Builds VGG16-based CNN model
3. Trains for up to 50 epochs (can stop early)
4. Saves best model to `trained_models/alzheimer_model_final.h5`
5. Generates performance visualizations

**Training Output Files:**
- `alzheimer_model_final.h5` - Trained model (30-50 MB)
- `model_metadata.pkl` - Class names and mappings
- `confusion_matrix.png` - Performance visualization
- `training_history.png` - Training/validation curves

### STEP 3: Use Your Trained Model

1. **Stop the backend server** (Ctrl+C in backend terminal)
2. **Restart the server:**
   ```bash
   python app.py
   ```
3. You should see:
   ```
   ✓ Model loaded successfully with compile=False!
   ✓ Metadata loaded!
   ```

Your trained model is now being used for predictions!

---

## 🔧 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
**GET** `/api/health`

**Response:**
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "tensorflow_version": "2.15.0"
}
```

#### 2. Predict Alzheimer's Stage
**POST** `/api/predict`

**Headers:**
```
Content-Type: multipart/form-data
```

**Body:**
```
file: <binary image data>
```

**Response (Success):**
```json
{
  "success": true,
  "diagnosis": "No Impairment",
  "stage": "Stage 0 - Healthy",
  "confidence": 0.87,
  "severity_score": 5,
  "probabilities": {
    "No Impairment": 0.87,
    "Very Mild Impairment": 0.08,
    "Mild Impairment": 0.03,
    "Moderate Impairment": 0.02
  },
  "recommendations": [
    "Maintain healthy lifestyle",
    "Regular physical exercise (30 min/day)",
    "Cognitive stimulation activities",
    "Balanced Mediterranean diet",
    "Annual health checkups"
  ]
}
```

**Response (Error):**
```json
{
  "success": false,
  "error": "Invalid file type. Use JPG, JPEG, or PNG"
}
```

---

## �️ Troubleshooting

### Backend Issues

#### Problem: "ModuleNotFoundError: No module named 'flask'"
**Solution:**
```bash
cd server
pip install -r requirements.txt
```

#### Problem: "Address already in use"
**Solution:** Port 5000 is occupied
```bash
# Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Mac/Linux
lsof -ti:5000 | xargs kill -9
```

#### Problem: "TensorFlow not available"
**This is OK!** The server will use Mock Model mode for demonstration.

To install TensorFlow (optional):
```bash
pip install tensorflow==2.15.0
```

---

### Frontend Issues

#### Problem: "npm: command not found"
**Solution:** Install Node.js from [nodejs.org](https://nodejs.org/)

#### Problem: "Port 3000 is already in use"
**Solution:**
```bash
# Kill the process on port 3000
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Or change the port in vite.config.js to 3001
```

#### Problem: "Cannot connect to server"
**Solution:**
1. Make sure backend is running on port 5000
2. Check browser console for CORS errors
3. Verify `http://localhost:5000/api/health` works

---

### Common Questions

**Q: Do I need a trained model to run the app?**
A: No! It works with Mock Model mode by default.

**Q: How do I stop the servers?**
A: Press `Ctrl+C` in each terminal window.

**Q: Can I use my own dataset?**
A: Yes! Follow the training section above.

**Q: Is this for clinical use?**
A: No, this is for research/educational purposes only.

---

## � Tech Stack

**Frontend:**
- React 19.2.0
- Vite 7.2.2 (build tool)
- Axios 1.13.2 (HTTP client)
- Custom CSS (no external UI libraries)

**Backend:**
- Flask 3.0.0 (web framework)
- OpenCV 4.9.0 (image processing)
- NumPy 1.26.4 (numerical computing)
- TensorFlow 2.15.0 (optional - for training)

**Model:**
- Architecture: VGG16 CNN
- Input: 176×208×3 RGB images
- Output: 4-class softmax
- Preprocessing: Histogram equalization, skull stripping, normalization

---

## ⚠️ Important Disclaimers

1. **NOT for Clinical Diagnosis**: This is a research/educational tool
2. **Consult Professionals**: Always seek medical advice from qualified professionals
3. **Mock Model**: Default mode uses random predictions for demonstration
4. **Training Required**: Real predictions require training with medical data

---

## 📄 License

Educational and research use only.

---

## 🆘 Need More Help?

1. Check the `walkthrough.md` file in the artifacts
2. Review error messages carefully
3. Ensure all prerequisites are installed
4. Verify both servers are running

---

## 👤 Author

Built for Alzheimer's Disease detection research and education.

**Last Updated:** November 2025
