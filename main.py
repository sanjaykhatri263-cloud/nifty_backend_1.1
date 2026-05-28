"""
Nifty Sniper — Live Signal Backend  v2
=======================================
New in v2: 
  • Stateful Memory Appending
  • 10-Day Warmup
  • Instant Historical Backtest Generator on Boot
"""

import asyncio
import json
import logging
import os
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from auth import (
    Token, UserIn, UserOut,
    add_subscriber, authenticate_user, change_password,
    create_token, delete_subscriber, get_current_user,
    list_subscribers, require_admin, update_subscriber_status,
)
from data_sources import DataSourceManager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("nifty")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent
SCALER_PATH = BASE / "models" / "nifty_scaler_dual_V2.pkl"
LONG_PATH   = BASE / "models" / "long_brain_10bar_V2.pth"
SHORT_PATH  = BASE / "models" / "short_brain_10bar_V2.pth"

# ── Hyperparams ───────────────────────────────────────────────────────────────
SEQ_LEN     = 10
INPUT_DIM   = 70
D_MODEL     = 128
N_HEAD      = 4
NUM_LAYERS  = 3
LONG_THRESH = 0.80   
SHORT_THRESH= 0.80
POLL_SECS   = 60

# ── Model ─────────────────────────────────────────────────────────────────────
class NiftySniperBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD,
                                       batch_first=True, dropout=0.1),
            num_layers=NUM_LAYERS
        )
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 64), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.fc(self.transformer(self.proj(x))[:, -1, :]).squeeze(-1)


# ── Features ──────────────────────────────────────────────────────────────────
def generate_blueprint_features(df_dict: dict) -> pd.DataFrame:
    master_df = df_dict["2m"][["Open", "High", "Low", "Close"]].copy()
    feature_matrix = pd.DataFrame(index=master_df.index)
    tfs = ["1m", "2m", "5m", "15m", "60m"]
    raw_data = {}
    for tf in tfs:
        df = df_dict[tf].copy()
        raw_data[tf] = {
            "Close": df["Close"], "High": df["High"], "Low": df["Low"],
            "H4": df["Close"].shift(1) + (df["High"].shift(1) - df["Low"].shift(1)) * 1.1 / 2,
            "L4": df["Close"].shift(1) - (df["High"].shift(1) - df["Low"].shift(1)) * 1.1 / 2,
            "RSI": ta.rsi(df["Close"], length=14),
            "ATR": ta.atr(df["High"], df["Low"], df["Close"], length=14),
            "MA18": ta.sma(df["Close"], length=18),
            "MA40": ta.sma(df["Close"], length=40),
        }
        macd_df = ta.macd(df["Close"], fast=12, slow=26, signal=9)
        adx_df  = ta.adx(df["High"], df["Low"], df["Close"], length=14)
        raw_data[tf]["MACD"]   = macd_df.iloc[:, 0] if macd_df is not None else None
        raw_data[tf]["Signal"] = macd_df.iloc[:, 1] if macd_df is not None else None
        raw_data[tf]["Hist"]   = macd_df.iloc[:, 2] if macd_df is not None else None
        raw_data[tf]["ADX"]    = adx_df.iloc[:, 0]  if adx_df  is not None else None
        raw_data[tf]["DIP"]    = adx_df.iloc[:, 1]  if adx_df  is not None else None
        raw_data[tf]["DIN"]    = adx_df.iloc[:, 2]  if adx_df  is not None else None

    def gs(tf, key):
        return raw_data[tf][key].reindex(master_df.index, method="ffill").shift(1)

    for tf in tfs:
        c, h4, l4, atr = gs(tf,"Close"), gs(tf,"H4"), gs(tf,"L4"), gs(tf,"ATR")
        feature_matrix[f"{tf}_Dist_To_H4"]   = (c - h4) / (atr + 1e-6)
        feature_matrix[f"{tf}_Dist_To_L4"]   = (c - l4) / (atr + 1e-6)
        feature_matrix[f"{tf}_Inside_H4_L4"] = ((c <= h4) & (c >= l4)).astype(float)
        rsi = gs(tf,"RSI")
        feature_matrix[f"{tf}_RSI"]        = rsi
        feature_matrix[f"{tf}_RSI_Ext_80"] = (rsi >= 80).astype(float)
        feature_matrix[f"{tf}_RSI_Ext_20"] = (rsi <= 20).astype(float)
        adx, dip, din = gs(tf,"ADX"), gs(tf,"DIP"), gs(tf,"DIN")
        feature_matrix[f"{tf}_ADX"]     = adx
        feature_matrix[f"{tf}_DI_Bull"] = (dip > din).astype(float)
        feature_matrix[f"{tf}_DI_Bear"] = (din > dip).astype(float)
        macd, hist = gs(tf,"MACD"), gs(tf,"Hist")
        feature_matrix[f"{tf}_MACD"]      = macd
        feature_matrix[f"{tf}_MACD_Hist"] = hist
        ma18, ma40 = gs(tf,"MA18"), gs(tf,"MA40")
        feature_matrix[f"{tf}_MA_Bull"]   = (ma18 > ma40).astype(float)
        feature_matrix[f"{tf}_Dist_MA18"] = (c - ma18) / (atr + 1e-6)
        feature_matrix[f"{tf}_MA_Spread"] = (ma18 - ma40) / (atr + 1e-6)
    return feature_matrix.ffill().bfill()


# ── Engine ────────────────────────────────────────────────────────────────────
class NiftySignalEngine:
    def __init__(self):
        self.device        = torch.device("cpu")
        self.scaler        = joblib.load(SCALER_PATH)
        self.long_brain    = self._load(LONG_PATH)
        self.short_brain   = self._load(SHORT_PATH)
        self.signal_history: deque = deque(maxlen=200)
        self.latest_signal: dict   = {}
        self.long_thresh   = LONG_THRESH
        self.short_thresh  = SHORT_THRESH
        self.data_mgr      = DataSourceManager()
        
        self.memory_df: Optional[pd.DataFrame] = None
        self.historical_predictions: pd.DataFrame = pd.DataFrame()
        
        log.info("Engine ready ✓")

    def _load(self, path):
        m = NiftySniperBrain()
        m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        m.eval()
        return m

    def _decide(self, pl, ps):
        if pl >= self.long_thresh  and pl > ps: return "BUY"
        if ps >= self.short_thresh and ps > pl: return "SELL"
        return "WAIT"

    def _apply_threshold(self, r: dict) -> dict:
        sig = self._decide(r["prob_long"] / 100, r["prob_short"] / 100)
        return {**r, "signal": sig,
                "long_thresh_pct":  round(self.long_thresh  * 100),
                "short_thresh_pct": round(self.short_thresh * 100)}

    def _warmup_and_backtest(self):
        """ Instantly processes the leftover valid days from the 10-day payload into backtest signals """
        log.info("Generating historical backtest signals for dashboard...")
        df_1m = self.memory_df.copy()
        
        freq_map = {"1m":"1min","2m":"2min","5m":"5min","15m":"15min","60m":"60min"}
        df_dict  = {
            tf: df_1m.resample(freq).agg(
                {"Open":"first","High":"max","Low":"min","Close":"last"}
            ).dropna()
            for tf, freq in freq_map.items()
        }
        
        if len(df_dict["2m"]) < SEQ_LEN + 5: return
        
        # This will drop the first ~7 days of NaN values caused by the 60m MA40
        fm = generate_blueprint_features(df_dict).dropna()
        if len(fm) < SEQ_LEN: return
        
        records = []
        # Loop over the remaining valid 3-4 days to generate simulated past signals
        for i in range(SEQ_LEN, len(fm) + 1):
            win = fm.iloc[i-SEQ_LEN:i].values.astype(np.float32)
            scaled = self.scaler.transform(win)
            x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pl = float(self.long_brain(x).item())
                ps = float(self.short_brain(x).item())
            
            bar_time = fm.index[i-1]
            price = float(df_dict["2m"].loc[bar_time, "Close"])
            
            records.append(pd.DataFrame({
                "Close": price, 
                "Long_Prob": pl, 
                "Short_Prob": ps
            }, index=[bar_time]))
        
        if records:
            self.historical_predictions = pd.concat(records)
            log.info(f"Backtest ready! {len(self.historical_predictions)} historical signals computed.")

    def run_inference(self) -> Optional[dict]:
        # 1. Stateful Data Fetching Logic (10 Days on boot, 15 Mins thereafter)
        if self.memory_df is None or self.memory_df.empty:
            log.info("Engine waking up: Fetching 10 days of history to warm up 60m indicators...")
            new_df = self.data_mgr.fetch_1m_ohlc(days=10, minutes=0)
            if new_df is None: return None
            self.memory_df = new_df
            
            # Instantly calculate the backtest
            self._warmup_and_backtest()
        else:
            log.info("Engine active: Fetching last 15 minutes of live ticks...")
            new_df = self.data_mgr.fetch_1m_ohlc(days=0, minutes=15)
            if new_df is None: return None
            
            # Append, remove duplicates, and sort
            merged = pd.concat([self.memory_df, new_df])
            merged = merged[~merged.index.duplicated(keep='last')].sort_index()
            
            # Keep memory capped at roughly 10 trading days
            self.memory_df = merged.tail(4000)

        # Work on a copy of the memory
        df_1m = self.memory_df.copy()
        
        freq_map = {"1m":"1min","2m":"2min","5m":"5min","15m":"15min","60m":"60min"}
        df_dict  = {
            tf: df_1m.resample(freq).agg(
                {"Open":"first","High":"max","Low":"min","Close":"last"}
            ).dropna()
            for tf, freq in freq_map.items()
        }
        
        if len(df_dict["2m"]) < SEQ_LEN + 5:
            log.warning("Not enough 2m bars after resample")
            return None
            
        fm = generate_blueprint_features(df_dict).dropna()
        if len(fm) < SEQ_LEN:
            return None
            
        win    = fm.iloc[-SEQ_LEN:].values.astype(np.float32)
        scaled = self.scaler.transform(win)
        x      = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pl = float(self.long_brain(x).item())
            ps = float(self.short_brain(x).item())
            
        signal = self._decide(pl, ps)
        price  = float(df_dict["2m"]["Close"].iloc[-1])
        bar    = df_dict["2m"].index[-1]
        
        # Save live prediction into memory for the historical endpoint
        if self.historical_predictions.empty or bar not in self.historical_predictions.index:
            new_pred = pd.DataFrame({"Close": price, "Long_Prob": pl, "Short_Prob": ps}, index=[bar])
            if self.historical_predictions.empty:
                self.historical_predictions = new_pred
            else:
                self.historical_predictions = pd.concat([self.historical_predictions, new_pred])
            
            # Cap backtest history at 2000 rows (approx 5-6 trading days)
            self.historical_predictions = self.historical_predictions.tail(2000) 
            
        result = {
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "bar_time":         str(bar),
            "price":            round(price, 2),
            "prob_long":        round(pl * 100, 1),
            "prob_short":       round(ps * 100, 1),
            "signal":           signal,
            "long_thresh_pct":  round(self.long_thresh  * 100),
            "short_thresh_pct": round(self.short_thresh * 100),
            "rsi_2m":           round(float(fm["2m_RSI"].iloc[-1]), 1),
            "adx_2m":           round(float(fm["2m_ADX"].iloc[-1]), 1),
            "ma_bull_2m":       int(fm["2m_MA_Bull"].iloc[-1]),
            "data_source":      self.data_mgr.current_source,
        }
        
        self.signal_history.appendleft(result)
        self.latest_signal = result
        log.info(f"{signal}  long={pl:.1%}  short={ps:.1%}  ₹{price:.2f}  src={self.data_mgr.current_source}")
        return result


# ── FastAPI ───────────────────────────────────────────────────────────────────
app    = FastAPI(title="Nifty Sniper API v2")
engine = NiftySignalEngine()

# connected WebSockets keyed by username
clients: dict[str, set[WebSocket]] = {}   

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


async def broadcast(payload: dict):
    msg  = json.dumps(payload)
    dead: list[tuple[str, WebSocket]] = []
    for uname, wset in clients.items():
        for ws in list(wset):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append((uname, ws))
    for uname, ws in dead:
        clients.get(uname, set()).discard(ws)


async def inference_loop():
    while True:
        result = await asyncio.get_event_loop().run_in_executor(None, engine.run_inference)
        if result:
            await broadcast({"type": "signal", "data": result})
        await asyncio.sleep(POLL_SECS)


@app.on_event("startup")
async def startup():
    asyncio.create_task(inference_loop())
    log.info("Inference loop started")


# ══════════════════════════════════════════════════════════════════════════════
# Auth endpoints
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/auth/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials or account suspended")
    token = create_token({"sub": user["username"], "role": user["role"]})
    return Token(
        access_token=token, token_type="bearer",
        role=user["role"], username=user["username"], name=user["name"]
    )

@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {k: v for k, v in user.items() if k != "hashed_pw"}


# ══════════════════════════════════════════════════════════════════════════════
# Admin — subscriber management
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/admin/subscribers", response_model=list[UserOut])
async def get_subscribers(admin: dict = Depends(require_admin)):
    return list_subscribers()

@app.post("/admin/subscribers", response_model=UserOut)
async def create_subscriber(data: UserIn, admin: dict = Depends(require_admin)):
    return add_subscriber(data)

class StatusUpdate(BaseModel):
    status: str   

@app.patch("/admin/subscribers/{username}", response_model=UserOut)
async def patch_subscriber(username: str, body: StatusUpdate, admin: dict = Depends(require_admin)):
    return update_subscriber_status(username, body.status)

@app.delete("/admin/subscribers/{username}")
async def remove_subscriber(username: str, admin: dict = Depends(require_admin)):
    delete_subscriber(username)
    return {"deleted": username}

class PasswordReset(BaseModel):
    new_password: str

@app.post("/admin/subscribers/{username}/reset-password")
async def reset_password(username: str, body: PasswordReset, admin: dict = Depends(require_admin)):
    change_password(username, body.new_password)
    return {"ok": True}


# ══════════════════════════════════════════════════════════════════════════════
# Admin — data source management & History
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/admin/data-source")
async def get_data_source(admin: dict = Depends(require_admin)):
    return engine.data_mgr.status

class DataSourceSwitch(BaseModel):
    source:        str            
    api_key:       Optional[str] = None
    api_secret:    Optional[str] = None
    session_token: Optional[str] = None

@app.post("/admin/data-source")
async def switch_data_source(body: DataSourceSwitch, admin: dict = Depends(require_admin)):
    if body.source == "yfinance":
        status = engine.data_mgr.switch_to_yfinance()
    elif body.source == "breeze":
        if not all([body.api_key, body.api_secret, body.session_token]):
            raise HTTPException(status_code=400,
                detail="api_key, api_secret, and session_token are required for Breeze")
        status = engine.data_mgr.switch_to_breeze(
            body.api_key, body.api_secret, body.session_token
        )
    else:
        raise HTTPException(status_code=400, detail="source must be 'yfinance' or 'breeze'")
        
    # Clear the engine memory so it fetches a clean 10-day history from the new source
    engine.memory_df = None
    engine.historical_predictions = pd.DataFrame()
    
    await broadcast({"type": "data_source_changed", "data": status})
    return status

@app.get("/admin/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    return {
        "connected_users":  sum(len(v) for v in clients.values()),
        "signal_count":     len(engine.signal_history),
        "latest_signal":    engine.latest_signal,
        "data_source":      engine.data_mgr.status,
        "thresholds": {
            "long":  round(engine.long_thresh  * 100),
            "short": round(engine.short_thresh * 100),
        },
    }

@app.get("/admin/history")
async def get_historical_signals(days_back: int = 1, admin: dict = Depends(require_admin)):
    """
    Returns historical price action and AI probabilities so the frontend 
    can display yesterday's action and simulated signals.
    """
    df = engine.historical_predictions
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="Engine memory is empty or still warming up.")
    
    try:
        # Filter for only standard market hours (09:15 to 15:30)
        df = df.between_time('09:15', '15:30')
        
        if days_back > 0:
            last_date = df.index[-1].date()
            start_date = last_date - timedelta(days=days_back)
            df = df.loc[str(start_date):]

        history_list = []
        for index, row in df.iterrows():
            history_list.append({
                "time": index.strftime("%Y-%m-%d %H:%M:%S"),
                "close": round(row.get("Close", 0), 2),
                "long_prob": round(row.get("Long_Prob", 0) * 100, 1),
                "short_prob": round(row.get("Short_Prob", 0) * 100, 1)
            })
            
        return {"status": "ok", "records": len(history_list), "data": history_list}
    except Exception as e:
        log.error(f"Failed to generate history: {e}")
        raise HTTPException(status_code=500, detail="Failed to process historical data")


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket — authenticated, role-aware
# ══════════════════════════════════════════════════════════════════════════════
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query(...)):
    from jose import JWTError, jwt as _jwt
    from auth import SECRET_KEY, ALGORITHM, _load_users

    try:
        payload  = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        users    = _load_users()
        user     = users.get(username)
        if not user or user["status"] != "active":
            await ws.close(code=4001)
            return
    except JWTError:
        await ws.close(code=4001)
        return

    await ws.accept()
    clients.setdefault(username, set()).add(ws)
    log.info(f"WS connect: {username} ({user['role']})  total_sessions={sum(len(v) for v in clients.values())}")

    if engine.latest_signal:
        await ws.send_text(json.dumps({"type": "signal",  "data": engine.latest_signal}))
    if engine.signal_history:
        await ws.send_text(json.dumps({"type": "history", "data": list(engine.signal_history)}))
    await ws.send_text(json.dumps({"type": "data_source", "data": engine.data_mgr.status}))

    try:
        while True:
            raw = await ws.receive_text()
            if raw in ("ping", ""):
                continue
            try:
                msg = json.loads(raw)
                if msg.get("type") == "set_threshold" and user["role"] == "admin":
                    lt = max(0.50, min(0.99, float(msg.get("long",  engine.long_thresh))))
                    st = max(0.50, min(0.99, float(msg.get("short", engine.short_thresh))))
                    engine.long_thresh  = lt
                    engine.short_thresh = st
                    log.info(f"Thresholds set by {username}: L={lt:.0%} S={st:.0%}")
                    if engine.latest_signal:
                        await broadcast({"type": "signal",
                                         "data": engine._apply_threshold(engine.latest_signal)})
            except (json.JSONDecodeError, ValueError):
                pass
    except WebSocketDisconnect:
        clients.get(username, set()).discard(ws)
        log.info(f"WS disconnect: {username}")


# ══════════════════════════════════════════════════════════════════════════════
# Public
# ══════════════════════════════════════════════════════════════════════════════
@app.get("/health")
def health():
    return {"status": "ok", "source": engine.data_mgr.current_source}
