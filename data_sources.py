"""
data_sources.py — Pluggable data source layer
==============================================
Supports:
  • yfinance   — free, ~15-min delay, no auth needed
  • breeze     — ICICI Direct Breeze API, real-time, requires API credentials

The active source is controlled via the admin API:
  POST /admin/data-source  {"source": "yfinance"}   or   {"source": "breeze", ...creds}

Breeze credentials needed:
  api_key        → from ICICI Direct developer portal
  api_secret     → from ICICI Direct developer portal
  session_token  → generated daily via Breeze login URL
                   (admin pastes it fresh each morning)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

log = logging.getLogger("nifty.data")

# ──────────────────────────────────────────────────────────────────────────────
# Base class
# ──────────────────────────────────────────────────────────────────────────────
class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        """Return DataFrame with columns [Open, High, Low, Close], DatetimeIndex (tz-naive), 1-min bars."""
        ...

    @property
    @abstractmethod
    def status(self) -> dict:
        """Return a dict describing the source and its connection state."""
        ...


# ──────────────────────────────────────────────────────────────────────────────
# 1. yfinance source
# ──────────────────────────────────────────────────────────────────────────────
class YFinanceSource(DataSource):
    name = "yfinance"
    TICKER = "^NSEI"

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        try:
            raw = yf.download(self.TICKER, period="1d", interval="1m",
                              progress=False, auto_adjust=True)
            if raw is None or raw.empty or len(raw) < 40:
                log.warning("yfinance: too few bars returned")
                return None
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            raw = raw[["Open", "High", "Low", "Close"]].dropna()
            raw.index = pd.to_datetime(raw.index)
            raw.index = (raw.index.tz_localize(None)
                         if raw.index.tzinfo is None
                         else raw.index.tz_convert(None))
            log.info(f"yfinance: fetched {len(raw)} 1m bars")
            return raw
        except Exception as e:
            log.error(f"yfinance fetch error: {e}")
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


# ──────────────────────────────────────────────────────────────────────────────
# 2. ICICI Direct Breeze source
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class BreezeCredentials:
    api_key:       str
    api_secret:    str
    session_token: str   # refresh daily from ICICI login URL
    connected:     bool  = False
    error:         str   = ""


class BreezeSource(DataSource):
    """
    ICICI Direct Breeze real-time data source.

    Breeze setup steps (do once):
      1. Register at https://api.icicidirect.com/
      2. Create app → get api_key + api_secret
      3. Each morning: visit login URL to get session_token
         URL format: https://api.icicidirect.com/apiuser/login?api_key=<YOUR_KEY>
         After login it redirects to your redirect_url with ?apisession=<TOKEN>
      4. Paste the token into admin panel → Data Source settings

    Install: pip install breeze-connect
    """
    name = "breeze"

    def __init__(self, creds: BreezeCredentials):
        self.creds  = creds
        self._breeze = None
        self._connect()

    def _connect(self):
        try:
            from breeze_connect import BreezeConnect   # type: ignore
            b = BreezeConnect(api_key=self.creds.api_key)
            b.generate_session(
                api_secret=self.creds.api_secret,
                session_token=self.creds.session_token,
            )
            self._breeze         = b
            self.creds.connected = True
            self.creds.error     = ""
            log.info("Breeze: connected ✓")
        except ImportError:
            self.creds.error     = "breeze-connect not installed. Run: pip install breeze-connect"
            self.creds.connected = False
            log.error(self.creds.error)
        except Exception as e:
            self.creds.error     = str(e)
            self.creds.connected = False
            log.error(f"Breeze connect failed: {e}")

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        if not self.creds.connected or self._breeze is None:
            log.error("Breeze not connected — cannot fetch data")
            return None
        try:
            now   = datetime.now()
            start = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
            end   = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            resp = self._breeze.get_historical_data_v2(
                interval="1minute",
                from_date=start,
                to_date=end,
                stock_code="NIFTY",
                exchange_code="NFO",
                product_type="options",   # use "futures" or "cash" depending on instrument
            )

            if not resp or resp.get("Status") != 200:
                log.warning(f"Breeze API error: {resp}")
                return None

            records = resp.get("Success", [])
            if not records:
                log.warning("Breeze: no records returned")
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
            df.index = df.index.tz_localize(None) if df.index.tzinfo else df.index
            log.info(f"Breeze: fetched {len(df)} 1m bars")
            return df.dropna()

        except Exception as e:
            log.error(f"Breeze fetch error: {e}")
            return None

    @property
    def status(self) -> dict:
        return {
            "source":      "breeze",
            "realtime":    True,
            "delay_note":  "Real-time (live tick data)",
            "connected":   self.creds.connected,
            "error":       self.creds.error,
            "credentials": {
                "api_key":    self.creds.api_key[:6] + "****" if self.creds.api_key else "",
                "session_ok": self.creds.connected,
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# Manager — holds the active source, switchable at runtime
# ──────────────────────────────────────────────────────────────────────────────
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
        log.info("Switched data source → yfinance")
        return self._source.status

    def switch_to_breeze(self, api_key: str, api_secret: str, session_token: str) -> dict:
        creds         = BreezeCredentials(api_key=api_key, api_secret=api_secret,
                                          session_token=session_token)
        self._source  = BreezeSource(creds)
        log.info(f"Switched data source → Breeze (connected={creds.connected})")
        return self._source.status
