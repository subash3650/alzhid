# 🚀 Deploy EVERYTHING to Render - Super Easy!

## ✅ **Best Option: Deploy Both Frontend + Backend on Render**

Instead of splitting between Vercel and Render, deploy EVERYTHING on Render using one `render.yaml` file!

### **Benefits:**
- ✅ Everything in one place
- ✅ Automatic URL linking (frontend knows backend URL automatically!)
- ✅ One configuration file
- ✅ Free tier for both services
- ✅ Automatic HTTPS for both

---

## 📝 **How to Deploy (Super Simple)**

### Step 1: Make Code Changes

You need to make **2 small code changes** first:

#### Change 1: Update `client/src/App.jsx` (Line 12)

**Find this:**
```javascript
const API_URL = 'http://localhost:5000';
```

**Replace with:**
```javascript
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';
```

#### Change 2: Update `server/app.py` (Lines 30-36)

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

#### Change 3: Update `server/app.py` (Line 417)

**Find this:**
```python
        app.run(debug=True, host='0.0.0.0', port=5000)
```

**Replace with:**
```python
        port = int(os.getenv('PORT', 5000))
        app.run(host='0.0.0.0', port=port)
```

---

### Step 2: Push to GitHub

```bash
git add .
git commit -m "Add Render deployment config"
git push
```

---

### Step 3: Deploy to Render

1. **Go to [render.com](https://render.com)** and sign up/login with GitHub

2. **Click "New +" → "Blueprint"** (not "Web Service"!)

3. **Connect your repository**: Select `alzhid`

4. **Apply the Blueprint**: 
   - Render will read your `render.yaml` file
   - It will create TWO services automatically:
     - `alzhid-backend` (Flask API)
     - `alzhid-frontend` (React app)

5. **Click "Apply"** and wait 5-10 minutes ⏳

6. **You'll get TWO URLs:**
   - Frontend: `https://alzhid-frontend.onrender.com` ← Visit this one!
   - Backend: `https://alzhid-backend.onrender.com` (used by frontend)

---

### Step 4: Update FRONTEND_URL

After deployment, you need to update the backend's FRONTEND_URL:

1. Go to Render Dashboard
2. Click on **alzhid-backend** service
3. Go to **Environment** tab
4. Update `FRONTEND_URL` to: `https://alzhid-frontend.onrender.com`
5. **Save** (will auto-redeploy)

---

## 🎉 **Done!**

Visit: `https://alzhid-frontend.onrender.com`

Upload an MRI image and test it!

---

## 📊 **What You Get:**

```
Frontend: https://alzhid-frontend.onrender.com
Backend:  https://alzhid-backend.onrender.com
```

- ✅ Both services on Render
- ✅ Automatic HTTPS
- ✅ Free tier
- ✅ Auto-deploy on git push
- ✅ Environment variables auto-configured

---

## ⚠️ **Important Notes:**

**Free Tier Limits:**
- Both services spin down after 15 min inactivity
- First load after sleep = 30-60 seconds
- 750 hours/month per service (enough for 24/7)

**Auto-Configuration:**
- Frontend automatically gets backend URL via `render.yaml`
- Backend gets frontend URL (update manually after first deploy)

---

## 🔧 **Troubleshooting:**

**"Cannot find module" error on frontend?**
→ Make sure Node version is compatible. Render uses Node 14 by default.

**CORS errors?**
→ Make sure you updated `FRONTEND_URL` in backend environment variables

**Backend not working?**
→ Check logs in Render dashboard → alzhid-backend → Logs

---

## 🚀 **Alternative: Vercel (Frontend) + Render (Backend)**

If you prefer Vercel for frontend, see: [vercel_deployment.md](file:///C:/Users/subas/.gemini/antigravity/brain/35a74e47-69cd-45ef-aa12-1bdedb3a1f29/vercel_deployment.md)

But deploying both on Render is **simpler** and **easier**!

---

**Ready?** Make the 3 code changes above, then push and deploy! 🎉
