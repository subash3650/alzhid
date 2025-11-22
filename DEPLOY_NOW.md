# Render Deployment - Quick Start

## I recommend: Render.com

**Why?** Free tier, perfect for Flask + Python ML apps, automatic HTTPS, easy deployment.

---

## Step 1: Make Code Changes

### Change 1: Update `server/app.py` (Line 30-36)

**Find this:**
```python
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

**Replace with:**
```python
# Get frontend URL from environment variable
FRONTEND_URL = os.getenv('FRONTEND_URL', 'http://localhost:3000')

CORS(app, resources={
    r"/api/*": {
        "origins": [FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
```

### Change 2: Update `server/app.py` (Line 417)

**Find this:**
```python
        app.run(debug=True, host='0.0.0.0', port=5000)
```

**Replace with:**
```python
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
```

### Change 3: Update `client/src/App.jsx` (Line 12)

**Find this:**
```javascript
const API_URL = 'http://localhost:5000';
```

**Replace with:**
```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
```

### Change 4: ✅ Already Done!
`server/requirements.txt` has been updated with:
- opencv-python-headless (for server deployment)
- gunicorn (production WSGI server)
- python-dotenv (environment variables)

---

## Step 2: Push Changes

```bash
cd a:\OneDrive\Desktop\alzhid
git add .
git commit -m "Prepare for Render deployment"
git push
```

---

## Step 3: Deploy Backend to Render

1. Go to **[render.com](https://render.com)** → Sign up/Login with GitHub
2. Click **"New +"** → **"Web Service"**
3. Connect your repository: `alzhid`
4. Configure:
   - **Name**: `alzhid-backend`
   - **Root Directory**: `server`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
   - **Instance Type**: `Free`

5. Click **"Advanced"** → Add Environment Variables:
   - `FRONTEND_URL` = `https://your-app.vercel.app` (your Vercel URL)
   - `FLASK_ENV` = `production`

6. Click **"Create Web Service"**
7. Wait 5-10 minutes ⏳
8. You'll get: `https://alzhid-backend.onrender.com`

---

## Step 4: Connect Frontend and Backend

### Update Vercel:
1. Go to Vercel Dashboard → Your Project
2. **Settings** → **Environment Variables**
3. Update `VITE_API_URL` = `https://alzhid-backend.onrender.com`
4. **Redeploy**

### Update Render (if you didn't add before):
1. Go to Render Dashboard → Your Service
2. **Environment** tab
3. Add `FRONTEND_URL` = `https://your-app.vercel.app`
4. Save (auto-redeploys)

---

## Step 5: Test

Visit your Vercel frontend → Upload MRI → Should work! 🎉

---

## ⚠️ Important Notes

- **Free tier**: Spins down after 15 min inactivity (first load = 30-60s)
- **opencv-python-headless**: Required for server deployment
- **CORS**: URLs must match EXACTLY (including https://)

---

## 🔧 Troubleshooting

**Build fails on Render?**
→ Check logs, ensure `opencv-python-headless` in requirements.txt

**CORS errors?**
→ Verify `FRONTEND_URL` in Render matches your Vercel URL exactly

**Can't connect?**
→ Test backend directly: `https://your-backend.onrender.com/api/health`

---

For full guide, see: [RENDER_DEPLOYMENT.md](file:///a:/OneDrive/Desktop/alzhid/RENDER_DEPLOYMENT.md)
