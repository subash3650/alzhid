# 🚀 Deploy Backend to Render - Step by Step

Complete guide to deploy your Flask backend to Render.com (Free & Recommended)

---

## Why Render?

- ✅ Free tier with 750 hours/month
- ✅ Perfect for Flask + Python ML apps
- ✅ Automatic HTTPS and SSL
- ✅ Auto-deploy from GitHub
- ✅ Environment variable support
- ✅ Better than Heroku (no free tier) and more reliable than Railway

---

## Step 1: Update Backend for Production

### 1.1 Update CORS in `app.py`

Edit `server/app.py` lines 30-36 to support both local and production:

```python
import os

# At the top, after other imports
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

# Update CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": [FRONTEND_URL],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

### 1.2 Update Flask Run Configuration

Edit `server/app.py` line 417:

**Change from:**
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

**To:**
```python
port = int(os.getenv('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

### 1.3 Add Production Dependencies

Add to `server/requirements.txt`:

```
flask==3.0.0
flask-cors==4.0.0
werkzeug==3.0.1
opencv-python-headless==4.9.0.80
numpy==1.26.4
joblib==1.3.2
gunicorn==21.2.0
python-dotenv==1.0.0
```

**Note:** Changed `opencv-python` to `opencv-python-headless` (required for server deployment without GUI)

### 1.4 Commit Changes

```bash
cd a:\OneDrive\Desktop\alzhid
git add .
git commit -m "Prepare backend for Render deployment"
git push
```

---

## Step 2: Deploy to Render

### 2.1 Sign Up

1. Go to [render.com](https://render.com)
2. Click **"Get Started"**
3. Sign up with GitHub (recommended)

### 2.2 Create New Web Service

1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repository
3. Select your repository: `alzhid`

### 2.3 Configure Web Service

Fill in these settings:

| Setting | Value |
|---------|-------|
| **Name** | `alzhid-backend` (or your choice) |
| **Region** | Choose closest to you |
| **Root Directory** | `server` |
| **Environment** | `Python 3` |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app` |
| **Instance Type** | `Free` |

### 2.4 Add Environment Variables

Click **"Advanced"** and add these environment variables:

| Key | Value |
|-----|-------|
| `FRONTEND_URL` | `https://your-app.vercel.app` (your Vercel URL) |
| `FLASK_ENV` | `production` |
| `PYTHON_VERSION` | `3.9.18` |

### 2.5 Deploy

1. Click **"Create Web Service"**
2. Wait 5-10 minutes for first deployment
3. You'll get a URL like: `https://alzhid-backend.onrender.com`

---

## Step 3: Test Your Backend

### 3.1 Test Health Endpoint

Visit: `https://your-app.onrender.com/api/health`

You should see:
```json
{
  "status": "healthy",
  "model_status": "loaded",
  "tensorflow_version": "Not Available"
}
```

### 3.2 Check Render Logs

1. Go to your Render dashboard
2. Click on your service
3. Go to **"Logs"** tab
4. You should see:
   ```
   🧠 Alzheimer Detection API Server
   TensorFlow: Not Available
   ⚠ USING MOCK MODEL - FOR DEMONSTRATION ONLY
   ✓ Ready to accept predictions
   Server running on: http://0.0.0.0:10000
   ```

---

## Step 4: Update Frontend

Update your Vercel environment variable:

1. Go to [vercel.com](https://vercel.com) → Your Project
2. Go to **Settings** → **Environment Variables**
3. Update or add:
   - **Name**: `VITE_API_URL`
   - **Value**: `https://your-app.onrender.com`
4. **Redeploy** your frontend

---

## Step 5: Test Full Application

1. Visit your Vercel frontend: `https://your-app.vercel.app`
2. Upload an MRI image
3. Click **"Analyze Image"**
4. Should work! 🎉

---

## 🔧 Troubleshooting

### Issue: Build Fails

**Error:** `opencv-python` installation fails

**Solution:** Make sure you're using `opencv-python-headless` in requirements.txt

---

### Issue: "Application Failed to Respond"

**Solution:** 
1. Check Render logs
2. Ensure start command is: `gunicorn app:app`
3. Verify `PORT` environment variable usage in `app.py`

---

### Issue: CORS Errors

**Solution:**
1. Verify `FRONTEND_URL` environment variable in Render matches your Vercel URL exactly
2. Include `https://` in the URL
3. Redeploy after changes

---

### Issue: Slow First Load (Cold Start)

**Info:** Render free tier spins down after 15 minutes of inactivity. First request after inactivity takes 30-60 seconds.

**Solutions:**
- Upgrade to paid tier ($7/month) for always-on
- Use a uptime monitor like [UptimeRobot](https://uptimerobot.com) (free) to ping every 5 minutes

---

## 🔄 Making Updates

### Update Backend Code

```bash
git add .
git commit -m "Update backend"
git push
```

Render will automatically redeploy! 🎉

### Update Environment Variables

1. Go to Render Dashboard → Your Service
2. Click **"Environment"** tab
3. Edit variables
4. Click **"Save Changes"**
5. Render will automatically redeploy

---

## 📊 Your Complete Deployment

After full deployment:

```
Frontend (Vercel):  https://your-app.vercel.app
Backend (Render):   https://alzhid-backend.onrender.com
```

**Environment Variables:**

**Vercel:**
- `VITE_API_URL` = `https://alzhid-backend.onrender.com`

**Render:**
- `FRONTEND_URL` = `https://your-app.vercel.app`
- `FLASK_ENV` = `production`

---

## 💰 Render Free Tier Limits

- ✅ 750 hours/month (enough for 1 service 24/7)
- ✅ 0.1 CPU
- ✅ 512 MB RAM
- ✅ Spins down after 15 min inactivity
- ✅ 100 GB bandwidth/month
- ✅ Automatic HTTPS

**Perfect for this project!**

---

## 🚀 Alternative Platforms

If Render doesn't work for you:

### Railway.app
- Similar to Render
- $5 credit/month free
- Good for Python apps
- Deploy command: `gunicorn app:app`

### Fly.io
- Generous free tier
- Requires Docker (more complex)
- Great performance

### PythonAnywhere
- Specifically for Python apps
- Free tier available
- Good for Flask apps

---

## 🎯 Next Steps

1. ✅ Deploy backend to Render
2. ✅ Get backend URL (e.g., `https://alzhid-backend.onrender.com`)
3. ✅ Update Vercel environment variable with backend URL
4. ✅ Update Render environment variable with Vercel URL
5. ✅ Test the application end-to-end
6. 🎉 Share your deployed app!

---

## ⚠️ Important Notes

> [!WARNING]
> **Cold Starts**: Free tier apps spin down after 15 minutes of inactivity. First request will be slow (30-60s).

> [!IMPORTANT]
> **opencv-python-headless**: Must use headless version for server deployment (no GUI dependencies).

> [!TIP]
> **Logs**: Always check Render logs if something doesn't work. They're very helpful for debugging!

---

**Need help?** Check [Render Documentation](https://render.com/docs) or the main deployment_guide.md
