
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import re
import requests
import streamlit.components.v1 as components

st.set_page_config(page_title="Advanced AI Investment Platform", layout="wide")
st.title("🧠 Advanced AI Investment Platform (Swing + Long-Term)")
st.caption("Decision-support dashboard (not financial advice). Multi-factor technical + fundamental scoring with transparent rationale.")

# ----------------------------
# Helpers
# ----------------------------
def normalize_symbol(user_input: str) -> str:
    s = (user_input or "").strip().upper()
    s = re.sub(r"\s+", "", s)
    return s

def safe_num(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def fmt_big(n):
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except Exception:
        return "N/A"
    absn = abs(n)
    if absn >= 1e12: return f"{n/1e12:.2f}T"
    if absn >= 1e9:  return f"{n/1e9:.2f}B"
    if absn >= 1e6:  return f"{n/1e6:.2f}M"
    if absn >= 1e3:  return f"{n/1e3:.2f}K"
    return f"{n:.0f}"

def score_bucket(score):
    if score >= 75: return "🟢 Strong"
    if score >= 55: return "🟡 Moderate"
    return "🔴 Weak"

# India shorthand map (extend anytime)
INDIA_ALIAS = {
    "HDFC": "HDFCBANK",
    "ICICI": "ICICIBANK",
    "KOTAK": "KOTAKBANK",
    "BAJAJFIN": "BAJFINANCE",
    "BAJAJFINSV": "BAJAJFINSV",
    "MM": "M&M",
    "M&M": "M&M",
    "LTI": "LTIM",
    "L&T": "LT",
    "LNT": "LT",
}

def build_candidates(user_symbol: str):
    s = normalize_symbol(user_symbol)
    if not s:
        return []
    base = INDIA_ALIAS.get(s, s)
    candidates = [base]
    if "." not in base:
        candidates += [f"{base}.NS", f"{base}.BO"]
    return candidates

def _ensure_ohlc(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.dropna()
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(set(df.columns)):
        return None
    return df

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_price_yahoo(symbol: str, period: str = "5y"):
    err1 = None
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        df = _ensure_ohlc(df)
        if df is not None and not df.empty:
            return df, None
    except Exception as e:
        err1 = str(e)
    if err1 is None:
        err1 = "No data from yf.download"

    try:
        t = yf.Ticker(symbol)
        df2 = t.history(period=period, interval="1d", auto_adjust=False)
        df2 = _ensure_ohlc(df2)
        if df2 is not None and not df2.empty:
            return df2, None
    except Exception as e:
        return None, f"download: {err1} | history: {e}"

    return None, f"download: {err1} | history: No data"

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_info_yahoo(symbol: str):
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        fin = getattr(t, "financials", None)
        bs = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        return info, fin, bs, cf
    except Exception:
        return {}, None, None, None

def _alpha_to_ohlc(js: dict):
    key = None
    for k in js.keys():
        if "Time Series" in k:
            key = k
            break
    if not key:
        return None
    rows = []
    for dt, vals in js[key].items():
        try:
            rows.append({
                "Date": pd.to_datetime(dt),
                "Open": float(vals.get("1. open")),
                "High": float(vals.get("2. high")),
                "Low": float(vals.get("3. low")),
                "Close": float(vals.get("4. close")),
                "Volume": float(vals.get("5. volume", 0)),
            })
        except Exception:
            continue
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("Date").set_index("Date").dropna()

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_price_alpha(symbol: str, api_key: str):
    url = "https://www.alphavantage.co/query"
    params = {"function": "TIME_SERIES_DAILY", "symbol": symbol, "apikey": api_key, "outputsize": "full"}
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    js = r.json()
    if "Error Message" in js:
        return None, js["Error Message"]
    if "Note" in js:
        return None, js["Note"]
    df = _alpha_to_ohlc(js)
    if df is None or df.empty:
        return None, "No data from Alpha Vantage"
    return df, None

def resolve_symbol_and_data(user_symbol: str, source: str, alpha_key):
    candidates = build_candidates(user_symbol)
    last_err = None
    for c in candidates:
        if source == "Yahoo Finance":
            d, e = fetch_price_yahoo(c, period="5y")
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
            last_err = e
        else:
            if not alpha_key:
                last_err = "Alpha Vantage API key missing"
                continue
            d, e = fetch_price_alpha(c, alpha_key)
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
            last_err = e
    return None, None, candidates, last_err

def build_google_finance_url(resolved_symbol: str) -> str:
    rs = resolved_symbol.upper()
    if rs.endswith(".NS"):
        base = rs[:-3]
        return f"https://www.google.com/finance/quote/{base}:NSE"
    if rs.endswith(".BO"):
        base = rs[:-3]
        return f"https://www.google.com/finance/quote/{base}:BOM"
    if ":" in rs:
        return f"https://www.google.com/finance/quote/{rs}"
    return f"https://www.google.com/finance/quote/{rs}:NASDAQ"

def build_tickertape_search_url(user_symbol: str) -> str:
    q = normalize_symbol(user_symbol)
    return f"https://www.tickertape.in/search?query={q}"

def iframe_block(title: str, url: str, height: int = 900, show_link: bool = False):
    st.markdown(f"### {title}")
    components.iframe(url, height=height, scrolling=True)
    if show_link:
        st.caption(url)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol_input = st.text_input("Script / Symbol", "RELIANCE")
    data_source = st.selectbox("Price data for indicators", ["Yahoo Finance", "Alpha Vantage (needs free API key)"])
    alpha_key = None
    if data_source.startswith("Alpha"):
        alpha_key = st.text_input("Alpha Vantage API Key (free)", type="password")
    min_rows = st.slider("Minimum history required (days)", 30, 400, 90)
    horizon = st.selectbox("Horizon", ["Swing (weeks)", "Long-term (months/years)"])
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"])
    st.divider()
    st.subheader("🔎 Research Panel (same page)")
    enable_embeds = st.checkbox("Show Google Finance + Tickertape inside app", value=True)
    show_embed_links = st.checkbox("Show embed URLs (optional)", value=False)
    st.caption("Some browsers block embedded sites (iframe restrictions).")

run = st.button("✅ Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter a symbol and click **Run Analysis**.")
    st.stop()

resolved, data, tried, err = resolve_symbol_and_data(symbol_input, data_source, alpha_key)

if data is None or resolved is None:
    st.error("Could not resolve symbol / insufficient data from selected source.")
    if err:
        st.warning(f"Provider message: {err}")
    with st.expander("What I tried"):
        st.write(tried)
    st.write("Try exact NSE ticker like RELIANCE.NS / TCS.NS / INFY.NS, or lower minimum history.")
    st.stop()

if len(data) < min_rows:
    st.warning(f"Only {len(data)} daily rows available for {resolved}. Analysis still runs, but confidence is lower.")

st.success(f"Using symbol: {resolved}  |  Rows: {len(data)}")

# Fundamentals (Yahoo only)
info, fin, bs, cf = ({}, None, None, None)
if data_source == "Yahoo Finance":
    info, fin, bs, cf = fetch_info_yahoo(resolved)

# Indicators
data = data.copy().dropna()
close = data["Close"].astype(float)
high = data["High"].astype(float)
low = data["Low"].astype(float)

def w_ok(w): 
    return len(data) >= (w + 5)

data["EMA20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator() if w_ok(20) else np.nan
data["EMA50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator() if w_ok(50) else np.nan
data["EMA200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator() if w_ok(200) else np.nan
data["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi() if w_ok(14) else np.nan

if w_ok(35):
    macd = ta.trend.MACD(close=close)
    data["MACD"] = macd.macd()
    data["MACD_SIGNAL"] = macd.macd_signal()
else:
    data["MACD"] = np.nan
    data["MACD_SIGNAL"] = np.nan

data["ADX14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx() if w_ok(14) else np.nan
data["ATR14"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range() if w_ok(14) else np.nan

latest = data.iloc[-1]
price = safe_num(latest.get("Close"))

def technical_score():
    reasons = []
    score = 50
    ema20, ema50, ema200 = safe_num(latest.get("EMA20")), safe_num(latest.get("EMA50")), safe_num(latest.get("EMA200"))
    rsi = safe_num(latest.get("RSI14"))
    macd_v, macd_s = safe_num(latest.get("MACD")), safe_num(latest.get("MACD_SIGNAL"))
    adx = safe_num(latest.get("ADX14"))

    if all(v is not None for v in [ema20, ema50, ema200, price]):
        if ema20 > ema50 > ema200 and price > ema20:
            score += 18; reasons.append("Uptrend (EMA20 > EMA50 > EMA200) and price above EMA20")
        elif ema20 < ema50 < ema200 and price < ema20:
            score -= 18; reasons.append("Downtrend (EMA20 < EMA50 < EMA200) and price below EMA20")
        else:
            reasons.append("Mixed trend (EMAs not aligned)")
    else:
        reasons.append("Trend score limited (not enough history for EMAs)")

    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 6; reasons.append(f"RSI healthy ({rsi:.1f})")
        elif rsi < 35:
            score += 4; reasons.append(f"RSI oversold ({rsi:.1f}) — potential bounce")
        elif rsi > 70:
            score -= 6; reasons.append(f"RSI overbought ({rsi:.1f}) — caution")

    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            score += 8; reasons.append("MACD above signal (bullish momentum)")
        else:
            score -= 6; reasons.append("MACD below signal (bearish momentum)")
    return int(max(0, min(100, score))), reasons

def fundamental_score():
    reasons = []
    score = 50
    if not info:
        reasons.append("Fundamentals limited (use Yahoo Finance source).")
        return score, reasons
    pe = safe_num(info.get("trailingPE"))
    roe = safe_num(info.get("returnOnEquity"))
    de = safe_num(info.get("debtToEquity"))
    margins = safe_num(info.get("profitMargins"))
    if pe is not None and pe > 0:
        if pe < 18: score += 12; reasons.append(f"Attractive P/E ({pe:.1f})")
        elif pe < 28: score += 6; reasons.append(f"Reasonable P/E ({pe:.1f})")
        else: score -= 6; reasons.append(f"High P/E ({pe:.1f})")
    if roe is not None:
        roe_pct = roe*100
        if roe_pct >= 15: score += 10; reasons.append(f"Strong ROE ({roe_pct:.1f}%)")
    if margins is not None:
        m = margins*100
        if m >= 12: score += 6; reasons.append(f"Good margins ({m:.1f}%)")
    if de is not None:
        if de > 180: score -= 8; reasons.append(f"High Debt/Equity ({de:.0f})")
    return int(max(0, min(100, score))), reasons

t_score, _ = technical_score()
f_score, _ = fundamental_score()
ai_score = int(round((0.70*t_score + 0.30*f_score) if horizon.startswith("Swing") else (0.45*t_score + 0.55*f_score)))

top1, top2, top3, top4 = st.columns([1.3, 1, 1, 1])
top1.metric("AI Score (0-100)", f"{ai_score}  {score_bucket(ai_score)}")
top2.metric("Technical Score", t_score)
top3.metric("Fundamental Score", f_score)
top4.metric("Last Close", f"{price:.2f}" if price is not None else "N/A")

# Embeds directly on the same page
if enable_embeds:
    st.subheader("🔎 Research Panel — Google Finance + Tickertape (same page)")
    gf_url = build_google_finance_url(resolved)
    tt_url = build_tickertape_search_url(symbol_input)

    left, right = st.columns(2)
    with left:
        iframe_block("Google Finance", gf_url, height=820, show_link=show_embed_links)
    with right:
        iframe_block("Tickertape (Search)", tt_url, height=820, show_link=show_embed_links)

    st.caption("If embeds appear blank, it's due to browser/iframe restrictions. You can turn off embeds in the sidebar.")

st.divider()

st.subheader("📉 Candlestick chart")
fig = go.Figure(data=[go.Candlestick(x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"])])
for name in ["EMA20", "EMA50", "EMA200"]:
    fig.add_trace(go.Scatter(x=data.index, y=data[name], mode="lines", name=name))
fig.update_layout(xaxis_rangeslider_visible=False, height=540)
st.plotly_chart(fig, use_container_width=True)

st.caption("Disclaimer: Analytics tool. Verify from official filings and multiple sources before investing.")
