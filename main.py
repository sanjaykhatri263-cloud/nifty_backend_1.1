"""
Nifty Sniper — Live Signal Backend  v3
=======================================
  • Stateful Memory Appending
  • Candlestick & Indicator Data Payload
  • Explicit IST (+05:30) Timezone Enforcement
  • Mode 1 (Repaint) vs Mode 2 (Strict) Global Toggle
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("nifty")

BASE        = Path(__file__).parent
SCALER_PATH = BASE / "models" / "nifty_scaler_dual_V2.pkl"
LONG_PATH   = BASE / "models" / "long_brain_10bar_V2.pth"
SHORT_PATH  = BASE / "models" / "short_brain_10bar_V2.pth"

SEQ_LEN     = 10
INPUT_DIM   = 70
D_MODEL     = 128
N_HEAD      = 4
NUM_LAYERS  = 3
LONG_THRESH = 0.80   
SHORT_THRESH= 0.80
POLL_SECS   = 60

class NiftySniperBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(INPUT_DIM, D_MODEL)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEAD, batch_first=True, dropout=0.1),
            num_layers=NUM_LAYERS
        )
        self.fc = nn.Sequential(nn.Linear(D_MODEL, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1), nn.Sigmoid())
    def forward(self, x):
        return self.fc(self.transformer(self.proj(x))[:, -1, :]).squeeze(-1)

def generate_blueprint_features(df_dict: dict):
    master_df = df_dict["2m"][["Open", "High", "Low", "Close"]].copy()
    feature_matrix = pd.DataFrame(index=master_df.index)
    raw_matrix = master_df.copy() 
    
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
        
        raw_matrix[f"{tf}_H4"] = h4
        raw_matrix[f"{tf}_L4"] = l4
        raw_matrix[f"{tf}_MACD"] = macd
        raw_matrix[f"{tf}_Hist"] = hist
        raw_matrix[f"{tf}_Signal"] = gs(tf, "Signal")

    return feature_matrix.ffill().bfill(), raw_matrix.ffill().bfill()


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
        
        # Admin Toggles
        self.eval_mode     = 1  # 1 = Classic (Repaint), 2 = Strict (No Repaint)
        
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
        return {**r, "signal": sig, "long_thresh_pct": round(self.long_thresh * 100), "short_thresh_pct": round(self.short_thresh * 100)}

    def _format_ist(self, dt_obj):
        """Forces output to strictly defined +05:30 ISO strings so frontend perfectly maps to IST."""
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        ist_time = dt_obj.astimezone(timezone(timedelta(hours=5, minutes=30)))
        return ist_time.strftime("%Y-%m-%dT%H:%M:%S+05:30")

    def _get_resampled_dict(self, df_1m):
        freq_map = {"1m":"1min","2m":"2min","5m":"5min","15m":"15min","60m":"60min"}
        if self.eval_mode == 1:
            return {tf: df_1m.resample(freq).agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna() for tf, freq in freq_map.items()}
        else:
            return {tf: df_1m.resample(freq, label='right', closed='right').agg({"Open":"first","High":"max","Low":"min","Close":"last"}).dropna() for tf, freq in freq_map.items()}

    def _warmup_and_backtest(self):
        log.info(f"Rebuilding history under Mode {self.eval_mode} rules...")
        if self.memory_df is None or self.memory_df.empty: return
        
        df_dict = self._get_resampled_dict(self.memory_df.copy())
        if len(df_dict["2m"]) < SEQ_LEN + 5: return
        
        fm, raw_fm = generate_blueprint_features(df_dict)
        fm = fm.dropna()
        raw_fm = raw_fm.loc[fm.index]
        
        # If Strict Mode, ignore the very last forming candle of the historical pull
        if self.eval_mode == 2 and len(fm) > SEQ_LEN:
            fm = fm.iloc[:-1]
            raw_fm = raw_fm.loc[fm.index]
            
        if len(fm) < SEQ_LEN: return
        
        records = []
        self.signal_history.clear()
        
        for i in range(SEQ_LEN, len(fm) + 1):
            win = fm.iloc[i-SEQ_LEN:i].values.astype(np.float32)
            scaled = self.scaler.transform(win)
            x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                pl = float(self.long_brain(x).item())
                ps = float(self.short_brain(x).item())
            
            bar_time = fm.index[i-1]
            r = raw_fm.loc[bar_time]
            price = float(r["Close"])
            ist_iso = self._format_ist(bar_time)
            
            records.append(pd.DataFrame({
                "Close": price, "Long_Prob": pl, "Short_Prob": ps,
            }, index=[bar_time]))
            
            result = {
                "timestamp":        ist_iso, "bar_time": ist_iso,
                "open": round(float(r["Open"]), 2), "high": round(float(r["High"]), 2), "low": round(float(r["Low"]), 2), "close": round(price, 2), "price": round(price, 2),
                "prob_long": round(pl * 100, 1), "prob_short": round(ps * 100, 1),
                "signal": self._decide(pl, ps), "long_thresh_pct": round(self.long_thresh * 100), "short_thresh_pct": round(self.short_thresh * 100),
                "rsi_2m": round(float(fm["2m_RSI"].iloc[i-1]), 1), "adx_2m": round(float(fm["2m_ADX"].iloc[i-1]), 1), "ma_bull_2m": int(fm["2m_MA_Bull"].iloc[i-1]),
                "h4_15m": round(float(r["15m_H4"]), 2), "l4_15m": round(float(r["15m_L4"]), 2),
                "h4_60m": round(float(r["60m_H4"]), 2), "l4_60m": round(float(r["60m_L4"]), 2),
                "macd": round(float(r["2m_MACD"]), 2), "macd_hist": round(float(r["2m_Hist"]), 2), "macd_signal": round(float(r["2m_Signal"]), 2),
                "data_source": self.data_mgr.current_source, "eval_mode": self.eval_mode
            }
            self.signal_history.appendleft(result)
            self.latest_signal = result
        
        if records:
            self.historical_predictions = pd.concat(records)
            log.info("Live buffer populated successfully.")

    def run_inference(self) -> Optional[dict]:
        if self.memory_df is None or self.memory_df.empty:
            new_df = self.data_mgr.fetch_1m_ohlc(days=15, minutes=0)
            if new_df is None: return None
            self.memory_df = new_df
            self._warmup_and_backtest()
        else:
            new_df = self.data_mgr.fetch_1m_ohlc(days=0, minutes=15)
            if new_df is None: return None
            merged = pd.concat([self.memory_df, new_df])
            self.memory_df = merged[~merged.index.duplicated(keep='last')].sort_index().tail(4000)

        df_dict = self._get_resampled_dict(self.memory_df.copy())
        if len(df_dict["2m"]) < SEQ_LEN + 5: return None
            
        fm, raw_fm = generate_blueprint_features(df_dict)
        fm = fm.dropna()
        
        # CRITICAL DIFFERENCE: Mode 1 evaluates open candle. Mode 2 evaluates closed candle.
        if self.eval_mode == 2 and len(fm) > SEQ_LEN:
            fm = fm.iloc[:-1]
            
        raw_fm = raw_fm.loc[fm.index]
        if len(fm) < SEQ_LEN: return None
            
        win    = fm.iloc[-SEQ_LEN:].values.astype(np.float32)
        scaled = self.scaler.transform(win)
        x      = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pl = float(self.long_brain(x).item())
            ps = float(self.short_brain(x).item())
            
        r       = raw_fm.iloc[-1]
        signal  = self._decide(pl, ps)
        price   = float(r["Close"])
        bar     = fm.index[-1]
        ist_iso = self._format_ist(bar)
        
        if self.historical_predictions.empty or bar not in self.historical_predictions.index:
            new_pred = pd.DataFrame({"Close": price, "Long_Prob": pl, "Short_Prob": ps}, index=[bar])
            if self.historical_predictions.empty: self.historical_predictions = new_pred
            else: self.historical_predictions = pd.concat([self.historical_predictions, new_pred])
            self.historical_predictions = self.historical_predictions.tail(2000) 
            
        result = {
            "timestamp":        ist_iso, "bar_time": ist_iso,
            "open": round(float(r["Open"]), 2), "high": round(float(r["High"]), 2), "low": round(float(r["Low"]), 2), "close": round(price, 2), "price": round(price, 2),
            "prob_long": round(pl * 100, 1), "prob_short": round(ps * 100, 1),
            "signal": signal, "long_thresh_pct": round(self.long_thresh * 100), "short_thresh_pct": round(self.short_thresh * 100),
            "rsi_2m": round(float(fm["2m_RSI"].iloc[-1]), 1), "adx_2m": round(float(fm["2m_ADX"].iloc[-1]), 1), "ma_bull_2m": int(fm["2m_MA_Bull"].iloc[-1]),
            "h4_15m": round(float(r["15m_H4"]), 2), "l4_15m": round(float(r["15m_L4"]), 2),
            "h4_60m": round(float(r["60m_H4"]), 2), "l4_60m": round(float(r["60m_L4"]), 2),
            "macd": round(float(r["2m_MACD"]), 2), "macd_hist": round(float(r["2m_Hist"]), 2), "macd_signal": round(float(r["2m_Signal"]), 2),
            "data_source": self.data_mgr.current_source, "eval_mode": self.eval_mode
        }
        
        self.signal_history.appendleft(result)
        self.latest_signal = result
        return result


app    = FastAPI(title="Nifty Sniper API v2")
engine = NiftySignalEngine()
clients: dict[str, set[WebSocket]] = {}   

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def broadcast(payload: dict):
    msg, dead = json.dumps(payload), []
    for uname, wset in clients.items():
        for ws in list(wset):
            try: await ws.send_text(msg)
            except: dead.append((uname, ws))
    for uname, ws in dead: clients.get(uname, set()).discard(ws)

async def inference_loop():
    while True:
        try:
            result = await asyncio.get_event_loop().run_in_executor(None, engine.run_inference)
            if result: await broadcast({"type": "signal", "data": result})
        except Exception as e: log.error(f"Engine Error: {e}")
        await asyncio.sleep(POLL_SECS)

@app.on_event("startup")
async def startup(): asyncio.create_task(inference_loop())

@app.post("/auth/token", response_model=Token)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user: raise HTTPException(status_code=401, detail="Invalid")
    return Token(access_token=create_token({"sub": user["username"], "role": user["role"]}), token_type="bearer", role=user["role"], username=user["username"], name=user["name"])

@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)): return {k: v for k, v in user.items() if k != "hashed_pw"}

@app.get("/admin/subscribers", response_model=list[UserOut])
async def get_subscribers(admin: dict = Depends(require_admin)): return list_subscribers()

@app.post("/admin/subscribers", response_model=UserOut)
async def create_subscriber(data: UserIn, admin: dict = Depends(require_admin)): return add_subscriber(data)

class StatusUpdate(BaseModel): status: str   

@app.patch("/admin/subscribers/{username}", response_model=UserOut)
async def patch_subscriber(username: str, body: StatusUpdate, admin: dict = Depends(require_admin)): return update_subscriber_status(username, body.status)

@app.delete("/admin/subscribers/{username}")
async def remove_subscriber(username: str, admin: dict = Depends(require_admin)):
    delete_subscriber(username)
    return {"deleted": username}

class PasswordReset(BaseModel): new_password: str

@app.post("/admin/subscribers/{username}/reset-password")
async def reset_password(username: str, body: PasswordReset, admin: dict = Depends(require_admin)):
    change_password(username, body.new_password)
    return {"ok": True}

@app.get("/admin/data-source")
async def get_data_source(admin: dict = Depends(require_admin)): return engine.data_mgr.status

class DataSourceSwitch(BaseModel):
    source: str            
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    session_token: Optional[str] = None

@app.post("/admin/data-source")
async def switch_data_source(body: DataSourceSwitch, admin: dict = Depends(require_admin)):
    if body.source == "yfinance": status = engine.data_mgr.switch_to_yfinance()
    elif body.source == "breeze": status = engine.data_mgr.switch_to_breeze(body.api_key, body.api_secret, body.session_token)
    else: raise HTTPException(status_code=400, detail="Invalid source")
        
    engine.memory_df = None
    engine.signal_history.clear()
    engine.latest_signal = {}
    await broadcast({"type": "data_source_changed", "data": status})
    return status

@app.get("/admin/stats")
async def admin_stats(admin: dict = Depends(require_admin)):
    return {"connected_users": sum(len(v) for v in clients.values()), "signal_count": len(engine.signal_history), "latest_signal": engine.latest_signal, "data_source": engine.data_mgr.status, "eval_mode": engine.eval_mode, "thresholds": {"long": round(engine.long_thresh * 100), "short": round(engine.short_thresh * 100)}}

@app.get("/admin/history")
async def get_historical_signals(days_back: int = 1, admin: dict = Depends(require_admin)):
    df = engine.historical_predictions
    if df is None or df.empty: raise HTTPException(status_code=400, detail="Engine empty.")
    try:
        df = df.between_time('09:15', '15:30')
        if days_back > 0:
            start_date = df.index[-1].date() - timedelta(days=days_back)
            df = df.loc[str(start_date):]

        history_list = []
        for index, row in df.iterrows():
            history_list.append({
                "time": engine._format_ist(index),
                "open": round(row.get("Open", 0), 2), "high": round(row.get("High", 0), 2), "low": round(row.get("Low", 0), 2), "close": round(row.get("Close", 0), 2), "price": round(row.get("Close", 0), 2),
                "long_prob": round(row.get("Long_Prob", 0) * 100, 1), "short_prob": round(row.get("Short_Prob", 0) * 100, 1),
                "h4_15m": round(row.get("H4_15m", 0), 2), "l4_15m": round(row.get("L4_15m", 0), 2), "h4_60m": round(row.get("H4_60m", 0), 2), "l4_60m": round(row.get("L4_60m", 0), 2),
            })
        return {"status": "ok", "records": len(history_list), "data": history_list}
    except: raise HTTPException(status_code=500, detail="History error")

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket, token: str = Query(...)):
    from jose import JWTError, jwt as _jwt
    from auth import SECRET_KEY, ALGORITHM, _load_users
    try:
        payload  = _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        users    = _load_users()
        user     = users.get(username)
        if not user or user["status"] != "active": return await ws.close(code=4001)
    except JWTError: return await ws.close(code=4001)

    await ws.accept()
    clients.setdefault(username, set()).add(ws)

    if engine.latest_signal: await ws.send_text(json.dumps({"type": "signal",  "data": engine.latest_signal}))
    if engine.signal_history: await ws.send_text(json.dumps({"type": "history", "data": list(engine.signal_history)}))
    await ws.send_text(json.dumps({"type": "data_source", "data": engine.data_mgr.status}))

    try:
        while True:
            raw = await ws.receive_text()
            if raw in ("ping", ""): continue
            try:
                msg = json.loads(raw)
                if msg.get("type") == "set_threshold" and user["role"] == "admin":
                    lt = max(0.50, min(0.99, float(msg.get("long",  engine.long_thresh))))
                    st = max(0.50, min(0.99, float(msg.get("short", engine.short_thresh))))
                    engine.long_thresh, engine.short_thresh = lt, st
                    if engine.latest_signal: await broadcast({"type": "signal", "data": engine._apply_threshold(engine.latest_signal)})
                
                # LISTENER FOR MODE TOGGLE
                if msg.get("type") == "set_eval_mode" and user["role"] == "admin":
                    new_mode = int(msg.get("mode", 1))
                    if engine.eval_mode != new_mode:
                        engine.eval_mode = new_mode
                        log.info(f"Admin switched to Mode {new_mode}. Recalculating charts...")
                        engine._warmup_and_backtest()
                        await broadcast({"type": "history", "data": list(engine.signal_history)})
                        await broadcast({"type": "eval_mode_changed", "mode": new_mode})
            except: pass
    except WebSocketDisconnect:
        clients.get(username, set()).discard(ws)

@app.get("/health")
def health(): return {"status": "ok"}
