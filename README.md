# Nifty Sniper — Backend Setup

## Local development

```bash
# 1. Place your model files in backend/models/
mkdir -p models
cp /path/to/nifty_scaler_dual_V2.pkl  models/
cp /path/to/long_brain_10bar_V2.pth   models/
cp /path/to/short_brain_10bar_V2.pth  models/

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run server (auto-reloads on save)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at http://localhost:8000
WebSocket at ws://localhost:8000/ws
Health check at http://localhost:8000/health

---

## Deploy to Railway (free)

1. Push the `backend/` folder to a GitHub repo
2. Go to https://railway.app → New Project → Deploy from GitHub
3. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Upload model files via Railway Volume or commit them (if <100 MB)
5. Copy the Railway public URL → update `VITE_WS_URL` in the frontend `.env`

---

## Deploy to Render (free)

1. Push backend to GitHub
2. New Web Service → connect repo
3. Build: `pip install -r requirements.txt`
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Free plan spins down after 15 min idle — use UptimeRobot to ping /health

---

## Environment variables (optional overrides)

| Variable     | Default | Description                  |
|-------------|---------|------------------------------|
| PORT        | 8000    | Port for uvicorn             |
| POLL_SECS   | 60      | Seconds between inferences   |
| LONG_THRESH | 0.60    | BUY threshold (0–1)          |
| SHORT_THRESH| 0.60    | SELL threshold (0–1)         |
