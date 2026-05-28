"""
data_sources.py — Fixed Breeze connection for Nifty 50 index cash data
Supports dynamic historical fetching (days) and live ticking (minutes).
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

log = logging.getLogger("nifty.data")


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_1m_ohlc(self, days: int = 0, minutes: int = 120) -> Optional[pd.DataFrame]:
        ...

    @property
    @abstractmethod
    def status(self) -> dict:
        ...


# ── yfinance (free, delayed, may fail on cloud) ───────────────────────────────
class YFinanceSource(DataSource):
    name = "yfinance"
    TICKER = "^NSEI"

    def fetch_1m_ohlc(self, days: int = 0, minutes: int = 120) -> Optional[pd.DataFrame]:
        try:
            import requests, yfinance as yf
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
            })
            ticker = yf.Ticker(self.TICKER, session=session)
            
            # yfinance uses strict periods. If days > 0 use 5d/1mo, else use 1d
            period = "1mo" if days > 5 else "5d" if days > 0 else "1d"
            raw    = ticker.history(period=period, interval="1m", auto_adjust=True)
            
            if raw is None or raw.empty or len(raw) < 5:
                log.warning("yfinance: too few bars")
                return None
            raw = raw[["Open","High","Low","Close"]].dropna()
            raw.index = pd.to_datetime(raw.index)
            raw.index = (raw.index.tz_localize(None)
                         if raw.index.tzinfo is None
                         else raw.index.tz_convert(None))
            log.info(f"yfinance OK: {len(raw)} bars fetched")
            return raw
        except Exception as e:
            log.warning(f"yfinance failed: {e}")
            return None

    @property
    def status(self) -> dict:
        return {
            "source":      "yfinance",
            "ticker":      self.TICKER,
            "realtime":    False,
            "delay_note":  "~15 min delay during market hours",
            "connected":   True,
            "credentials": None,
        }


# ── ICICI Direct Breeze ──────────────────────────────────────────────────────
@dataclass
class BreezeCredentials:
    api_key:       str
    api_secret:    str
    session_token: str
    connected:     bool = False
    error:         str  = ""


class BreezeSource(DataSource):
    name = "breeze"

    def __init__(self, creds: BreezeCredentials):
        self.creds   = creds
        self._breeze = None
        self._connect()

    def _connect(self):
        try:
            from breeze_connect import BreezeConnect
            b = BreezeConnect(api_key=self.creds.api_key)
            b.generate_session(
                api_secret=self.creds.api_secret,
                session_token=self.creds.session_token,
            )
            self._breeze         = b
            self.creds.connected = True
            self.creds.error     = ""
            log.info("Breeze connected ✓")
        except ImportError as e:
            self.creds.error     = f"Breeze Library Crash: {e}"
            self.creds.connected = False
            log.error(self.creds.error)
        except Exception as e:
            self.creds.error     = str(e)
            self.creds.connected = False
            log.error(f"Breeze connect failed: {e}")

    def fetch_1m_ohlc(self, days: int = 0, minutes: int = 120) -> Optional[pd.DataFrame]:
        if not self.creds.connected or self._breeze is None:
            log.error(f"Breeze not connected: {self.creds.error}")
            return None
        try:
            now = datetime.utcnow()
            
            # Dynamic Time Strategy (Pure UTC)
            if days > 0:
                start = (now - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            else:
                start = (now - timedelta(minutes=minutes)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
                
            end = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            log.info(f"Breeze fetching NIFTY from {start} to {end}")

            resp = self._breeze.get_historical_data_v2(
                interval="1minute",
                from_date=start,
                to_date=end,
                stock_code="NIFTY",       
                exchange_code="NSE",
                product_type="cash",
            )

            if not resp or resp.get("Status") != 200:
                log.error(f"Breeze error: {resp.get('Error') if resp else 'Empty'}")
                return None

            records = resp.get("Success", [])
            if not records:
                log.warning("Breeze: Status 200 but no records — market may be closed")
                return None

            df = pd.DataFrame(records)
            df = df.rename(columns={
                "datetime": "Datetime",
                "open":     "Open",
                "high":     "High",
                "low":      "Low",
                "close":    "Close",
            })

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")[["Open","High","Low","Close"]].sort_index()
            df = df.astype(float)
            df.index = (df.index.tz_localize(None)
                        if df.index.tzinfo is not None
                        else df.index)
            df = df.dropna()

            log.info(f"Breeze OK: {len(df)} bars fetched | latest: {df.index[-1]}")
            return df

        except Exception as e:
            log.error(f"Breeze fetch error: {e}", exc_info=True)
            return None

    def test_connection(self) -> dict:
        if not self.creds.connected or self._breeze is None:
            return {"ok": False, "error": self.creds.error}
        try:
            now   = datetime.utcnow()
            start = (now - timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end   = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            resp  = self._breeze.get_historical_data_v2(
                interval="1minute",
                from_date=start,
                to_date=end,
                stock_code="NIFTY",
                exchange_code="NSE",
                product_type="cash",
            )
            return {
                "ok":      resp.get("Status") == 200 if resp else False,
                "status":  resp.get("Status") if resp else None,
                "error":   resp.get("Error") if resp else "No response",
                "records": len(resp.get("Success", [])) if resp else 0,
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @property
    def status(self) -> dict:
        return {
            "source":      "breeze",
            "realtime":    True,
            "delay_note":  "Real-time data",
            "connected":   self.creds.connected,
            "error":       self.creds.error,
            "credentials": {
                "api_key":    self.creds.api_key[:6] + "****" if self.creds.api_key else "",
                "session_ok": self.creds.connected,
            },
        }


# ── Manager ───────────────────────────────────────────────────────────────────
class DataSourceManager:
    def __init__(self):
        self._source: DataSource = YFinanceSource()

    def fetch_1m_ohlc(self, days: int = 0, minutes: int = 120) -> Optional[pd.DataFrame]:
        return self._source.fetch_1m_ohlc(days, minutes)

    @property
    def status(self) -> dict:
        return self._source.status

    @property
    def current_source(self) -> str:
        return self._source.name

    def switch_to_yfinance(self) -> dict:
        self._source = YFinanceSource()
        log.info("Switched → yfinance")
        return self._source.status

    def switch_to_breeze(self, api_key: str, api_secret: str,
                         session_token: str) -> dict:
        creds        = BreezeCredentials(
            api_key=api_key,
            api_secret=api_secret,
            session_token=session_token,
        )
        self._source = BreezeSource(creds)
        log.info(f"Breeze switch result: connected={creds.connected} error={creds.error}")
        return self._source.status
