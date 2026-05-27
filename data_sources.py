"""
data_sources.py — Pluggable data source layer
==============================================
Supports:
  • yfinance / NSE direct — free, cloud-safe fallback chain
  • breeze               — ICICI Direct Breeze API, real-time
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Optional

import pandas as pd

log = logging.getLogger("nifty.data")

# ── Base ──────────────────────────────────────────────────────────────────────
class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        ...

    @property
    @abstractmethod
    def status(self) -> dict:
        ...


# ── yfinance (with cloud-safe fallback chain) ─────────────────────────────────
class YFinanceSource(DataSource):
    name = "yfinance"
    TICKER = "^NSEI"

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        return self._try_yfinance() or self._try_nse_direct()

    def _try_yfinance(self) -> Optional[pd.DataFrame]:
        try:
            import requests, yfinance as yf
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
                "Accept-Language": "en-US,en;q=0.9",
            })
            ticker = yf.Ticker(self.TICKER, session=session)
            raw    = ticker.history(period="1d", interval="1m", auto_adjust=True)
            if raw is None or raw.empty or len(raw) < 10:
                log.warning("yfinance: too few bars")
                return None
            raw = raw[["Open", "High", "Low", "Close"]].dropna()
            raw.index = pd.to_datetime(raw.index)
            raw.index = raw.index.tz_localize(None) if raw.index.tzinfo is None else raw.index.tz_convert(None)
            log.info(f"yfinance OK: {len(raw)} bars")
            return raw
        except Exception as e:
            log.warning(f"yfinance failed: {e}")
            return None

    def _try_nse_direct(self) -> Optional[pd.DataFrame]:
        """NSE India public chart API — works from cloud servers."""
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept":   "*/*",
                "Referer":  "https://www.nseindia.com/",
            })
            session.get("https://www.nseindia.com", timeout=10)
            url  = "https://www.nseindia.com/api/chart-databyindex?index=NIFTY+50&indices=true"
            resp = session.get(url, timeout=15)
            if resp.status_code != 200:
                log.warning(f"NSE API status {resp.status_code}")
                return None
            gd = resp.json().get("grapthData") or resp.json().get("graphData") or []
            if not gd:
                log.warning("NSE API: empty data")
                return None
            rows = []
            for item in gd:
                ts = pd.Timestamp(item[0], unit="ms", tz="Asia/Kolkata").tz_convert(None)
                p  = float(item[1])
                rows.append({"timestamp": ts, "Close": p, "Open": p, "High": p, "Low": p})
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            df = df["Close"].resample("1min").ohlc().dropna()
            df.columns = ["Open", "High", "Low", "Close"]
            log.info(f"NSE direct OK: {len(df)} bars")
            return df
        except Exception as e:
            log.warning(f"NSE direct failed: {e}")
            return None

    @property
    def status(self) -> dict:
        return {
            "source":     "yfinance",
            "ticker":     self.TICKER,
            "realtime":   False,
            "delay_note": "~15 min delay during market hours",
            "connected":  True,
            "credentials": None,
        }


# ── ICICI Direct Breeze ───────────────────────────────────────────────────────
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
            b.generate_session(api_secret=self.creds.api_secret,
                               session_token=self.creds.session_token)
            self._breeze         = b
            self.creds.connected = True
            self.creds.error     = ""
            log.info("Breeze connected ✓")
        except ImportError:
            self.creds.error     = "breeze-connect not installed"
            self.creds.connected = False
        except Exception as e:
            self.creds.error     = str(e)
            self.creds.connected = False
            log.error(f"Breeze connect failed: {e}")

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        if not self.creds.connected or self._breeze is None:
            return None
        try:
            now   = datetime.now()
            start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end   = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            resp  = self._breeze.get_historical_data_v2(
                interval="1minute", from_date=start, to_date=end,
                stock_code="NIFTY", exchange_code="NFO", product_type="options",
            )
            if not resp or resp.get("Status") != 200:
                return None
            records = resp.get("Success", [])
            if not records:
                return None
            df = pd.DataFrame(records).rename(columns={
                "datetime":"Datetime","open":"Open","high":"High","low":"Low","close":"Close"
            })
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")[["Open","High","Low","Close"]].sort_index()
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            return df.dropna()
        except Exception as e:
            log.error(f"Breeze fetch error: {e}")
            return None

    @property
    def status(self) -> dict:
        return {
            "source":      "breeze",
            "realtime":    True,
            "delay_note":  "Real-time tick data",
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

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        return self._source.fetch_1m_ohlc()

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

    def switch_to_breeze(self, api_key: str, api_secret: str, session_token: str) -> dict:
        creds        = BreezeCredentials(api_key=api_key, api_secret=api_secret,
                                         session_token=session_token)
        self._source = BreezeSource(creds)
        return self._source.status
