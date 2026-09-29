import logging
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import io

logger = logging.getLogger(__name__)

SYMBOL_MAP = {
    "SPX": "^SPX",
    "NDX": "^NDX",
    "VIX": "^VIX",
    "RUT": "^RUT",
    "DJI": "^DJI",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "GOLD": "GC=F",
    "OIL": "CL=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X"
}

def normalize_symbol(symbol: str) -> str:
    """Normalize input ticker symbol to Yahoo Finance standard."""
    s = str(symbol).strip().upper()
    return SYMBOL_MAP.get(s, s)

def fetch_historical_prices(symbol: str, years_back: int = 30) -> pd.DataFrame:
    """
    Fetch multi-decade daily/monthly OHLC prices with Yahoo Finance and Stooq fallback.
    Returns monthly resampled OHLC DataFrame.
    """
    norm_sym = normalize_symbol(symbol)
    start_date = (datetime.date.today() - datetime.timedelta(days=years_back * 365 + 100)).strftime("%Y-%m-%d")
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    
    df = pd.DataFrame()
    
    # 1. Primary: Yahoo Finance
    try:
        logger.info(f"Fetching {norm_sym} historical data from Yahoo Finance (Start: {start_date})...")
        ticker_obj = yf.Ticker(norm_sym)
        raw_df = ticker_obj.history(start=start_date, end=end_date, auto_adjust=True)
        if raw_df is not None and not raw_df.empty:
            df = raw_df.copy()
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
    except Exception as e:
        logger.warning(f"Yahoo Finance fetch failed for {norm_sym}: {e}")

    # 2. Secondary Fallback: Stooq CSV
    if df.empty or len(df) < 50:
        try:
            stooq_sym = norm_sym.lower().replace("^", "")
            if not stooq_sym.endswith(".us") and not "." in stooq_sym and len(stooq_sym) <= 5:
                stooq_sym = f"{stooq_sym}.us"
            stooq_url = f"https://stooq.com/q/d/l/?s={stooq_sym}&i=d"
            logger.info(f"Attempting Stooq fallback for {norm_sym} via {stooq_url}...")
            resp = requests.get(stooq_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and "Date,Open,High,Low,Close" in resp.text:
                stooq_df = pd.read_csv(io.StringIO(resp.text), parse_dates=["Date"], index_col="Date")
                stooq_df = stooq_df.sort_index()
                if not stooq_df.empty and len(stooq_df) > 50:
                    df = stooq_df
                    logger.info(f"Successfully retrieved {len(df)} records from Stooq for {norm_sym}")
        except Exception as e:
            logger.warning(f"Stooq fallback failed for {norm_sym}: {e}")

    if df.empty:
        raise ValueError(f"No price history found for symbol '{symbol}' across all data providers.")

    # Standardize columns
    df_clean = pd.DataFrame(index=df.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col in df.columns:
            df_clean[col] = df[col]
        elif col.lower() in [c.lower() for c in df.columns]:
            match_col = [c for c in df.columns if c.lower() == col.lower()][0]
            df_clean[col] = df[match_col]

    if 'Close' not in df_clean.columns:
        df_clean['Close'] = df.iloc[:, 0]
    if 'Open' not in df_clean.columns:
        df_clean['Open'] = df_clean['Close']
    if 'High' not in df_clean.columns:
        df_clean['High'] = df_clean['Close']
    if 'Low' not in df_clean.columns:
        df_clean['Low'] = df_clean['Close']

    df_clean = df_clean.dropna(subset=['Close'])
    df_clean = df_clean[df_clean['Close'] > 0]
    df_clean = df_clean.sort_index()

    # Resample to Monthly OHLC
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last'
    }
    
    # Use 'ME' if supported, else 'M'
    freq = 'ME' if hasattr(pd, 'DateOffset') else 'M'
    df_monthly = df_clean.resample(freq).agg(agg_rules).dropna()
    
    return df_monthly
