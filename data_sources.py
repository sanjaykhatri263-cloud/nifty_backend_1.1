"""
data_sources.py — Pluggable data source layer
Fallback chain: yfinance → NSE India → Yahoo CSV → Fake/demo data
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from io import StringIO

import pandas as pd

log = logging.getLogger("nifty.data")


class DataSource(ABC):
    name: str = "base"

    @abstractmethod
    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        ...

    @property
    @abstractmethod
    def status(self) -> dict:
        ...


class YFinanceSource(DataSource):
    name = "yfinance"
    TICKER = "^NSEI"

    def fetch_1m_ohlc(self) -> Optional[pd.DataFrame]:
        result = (
            self._try_yfinance_session()
            or self._try_yahoo_csv()
            or self._try_nse_api()
            or self._try_nse_chart()
        )
        if result is None:
            log.error("ALL data sources failed — no market data available")
        return result

    # ── Method 1: yfinance with spoofed browser session ──────────────────────
    def _try_yfinance_session(self) -> Optional[pd.DataFrame]:
        try:
            import requests
            import yfinance as yf

            session = requests.Session()
            session.headers.update({
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            })
            # Pre-warm cookies
            session.get("https://finance.yahoo.com", timeout=8)
            time.sleep(0.5)

            ticker = yf.Ticker(self.TICKER, session=session)
            raw    = ticker.history(period="1d", interval="1m", auto_adjust=True)
            if raw is None or raw.empty or len(raw) < 10:
                log.warning("Method1 yfinance: too few bars")
                return None
            raw = raw[["Open", "High", "Low", "Close"]].dropna()
            raw.index = pd.to_datetime(raw.index)
            raw.index = (raw.index.tz_localize(None)
                         if raw.index.tzinfo is None
                         else raw.index.tz_convert(None))
            log.info(f"Method1 yfinance OK: {len(raw)} bars")
            return raw
        except Exception as e:
            log.warning(f"Method1 yfinance failed: {e}")
            return None

    # ── Method 2: Yahoo Finance CSV endpoint (different from JSON API) ───────
    def _try_yahoo_csv(self) -> Optional[pd.DataFrame]:
        try:
            import requests
            now   = int(time.time())
            start = now - 86400  # last 24h
            url   = (
                f"https://query1.finance.yahoo.com/v7/finance/download/%5ENSEI"
                f"?period1={start}&period2={now}&interval=1m&events=history"
            )
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
                "Accept": "text/csv,application/csv",
            }
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code != 200 or "Date" not in resp.text[:100]:
                log.warning(f"Method2 Yahoo CSV: status {resp.status_code}")
                return None
            df = pd.read_csv(StringIO(resp.text), parse_dates=["Date"])
            df = df.rename(columns={"Date": "Datetime", "Open": "Open",
                                     "High": "High", "Low": "Low", "Close": "Close"})
            df = df.set_index("Datetime")[["Open", "High", "Low", "Close"]].dropna()
            df.index = pd.to_datetime(df.index)
            df.index = (df.index.tz_localize(None)
                        if df.index.tzinfo is None
                        else df.index.tz_convert(None))
            log.info(f"Method2 Yahoo CSV OK: {len(df)} bars")
            return df
        except Exception as e:
            log.warning(f"Method2 Yahoo CSV failed: {e}")
            return None

    # ── Method 3: NSE India official API ─────────────────────────────────────
    def _try_nse_api(self) -> Optional[pd.DataFrame]:
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
                "Accept": "*/*",
                "Referer": "https://www.nseindia.com/",
                "X-Requested-With": "XMLHttpRequest",
            })
            # Must get cookies first
            session.get("https://www.nseindia.com/market-data/live-equity-market",
                        timeout=10)
            time.sleep(1)
            resp = session.get(
                "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050",
                timeout=15
            )
            if resp.status_code != 200:
                log.warning(f"Method3 NSE API: status {resp.status_code}")
                return None
            data  = resp.json()
            price = float(data["data"][0]["lastPrice"].replace(",", ""))
            # Build a synthetic single bar from current price
            now = datetime.now().replace(second=0, microsecond=0)
            df  = pd.DataFrame([{
                "Open": price, "High": price, "Low": price, "Close": price
            }], index=[now])
            log.info(f"Method3 NSE API OK: price={price}")
            return df
        except Exception as e:
            log.warning(f"Method3 NSE API failed: {e}")
            return None

    # ── Method 4: NSE chart data endpoint ────────────────────────────────────
    def _try_nse_chart(self) -> Optional[pd.DataFrame]:
        try:
            import requests
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0",
                "Accept": "*/*",
                "Referer": "https://www.nseindia.com/",
            })
            session.get("https://www.nseindia.com", timeout=8)
            time.sleep(0.8)
            resp = session.get(
                "https://www.nseindia.com/api/chart-databyindex"
                "?index=NIFTY+50&indices=true",
                timeout=15
            )
            if resp.status_code != 200:
                log.warning(f"Method4 NSE chart: status {resp.status_code}")
                return None
            raw_data = resp.json()
            gd = raw_data.get("grapthData") or raw_data.get("graphData") or []
            if not gd:
                log.warning("Method4 NSE chart: empty graphData")
                return None
            rows = []
            for item in gd:
                ts = pd.Timestamp(item[0], unit="ms", tz="Asia/Kolkata").tz_convert(None)
                p  = float(item[1])
                rows.append({"timestamp": ts, "price": p})
            df = pd.DataFrame(rows).set_index("timestamp").sort_index()
            ohlc = df["price"].resample("1min").ohlc().dropna()
            ohlc.columns = ["Open", "High", "Low", "Close"]
            log.info(f"Method4 NSE chart OK: {len(ohlc)} bars")
            return ohlc
        except Exception as e:
            log.warning(f"Method4 NSE chart failed: {e}")
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
                "datetime": "Datetime", "open": "Open",
                "high": "High", "low": "Low", "close": "Close"
            })
            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df = df.set_index("Datetime")[["Open","High","Low","Close"]].sort_index()
            df.index = (df.index.tz_localize(None)
                        if df.index.tzinfo else df.index)
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

    def switch_to_breeze(self, api_key: str, api_secret: str,
                         session_token: str) -> dict:
        creds        = BreezeCredentials(api_key=api_key, api_secret=api_secret,
                                         session_token=session_token)
        self._source = BreezeSource(creds)
        return self._source.status
