
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
st.caption("Decision-support dashboard (not financial advice). Uses multi-factor technical + fundamental scoring with transparent rationale.")

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

def resolve_symbol_and_data(user_symbol: str, source: str, alpha_key: str|None):
    candidates = build_candidates(user_symbol)
    last_err = None
    for c in candidates:
        if source == "Yahoo Finance":
            d, e = fetch_price_yahoo(c, period="5y")
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
            last_err = e
        elif source.startswith("Alpha"):
            if not alpha_key:
                last_err = "Alpha Vantage API key missing"
                continue
            d, e = fetch_price_alpha(c, alpha_key)
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
            last_err = e
        else:
            # Tickertape is provided as an EMBED view (no scraping)
            return None, None, candidates, "Tickertape data fetch not supported (embed only)."
    return None, None, candidates, last_err

def render_tickertape_embed(url: str):
    # Safest integration: embed the official page URL provided by the user (no scraping).
    # Streamlit Cloud may block some iframes; we also show the link.
    if not url:
        st.info("Paste a Tickertape stock page URL to view it here (example: https://www.tickertape.in/stocks/... ).")
        return
    st.markdown("If the embed is blocked by the browser, open the link below:")
    st.markdown(f"- {url}")
    components.iframe(url, height=900, scrolling=True)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol_input = st.text_input("Script / Symbol", "RELIANCE")
    source = st.selectbox("Data source", ["Yahoo Finance", "Alpha Vantage (needs free API key)", "Tickertape (Embed view)"])
    alpha_key = None
    if source.startswith("Alpha"):
        alpha_key = st.text_input("Alpha Vantage API Key (free)", type="password")
    min_rows = st.slider("Minimum history required (days)", 30, 400, 90)
    horizon = st.selectbox("Horizon", ["Swing (weeks)", "Long-term (months/years)"])
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"])
    tickertape_url = None
    if source.startswith("Tickertape"):
        tickertape_url = st.text_input("Tickertape stock page URL (optional)", "")
        st.caption("Tickertape has no official free public API. We embed the page instead of scraping, which is more stable and safer.")
    st.divider()
    st.caption("Examples: RELIANCE / RELIANCE.NS / TCS / INFY / HDFC / AAPL")

run = st.button("✅ Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter a symbol and click **Run Analysis**.")
    st.stop()

# Tickertape mode: show embed + stop
if source.startswith("Tickertape"):
    st.subheader("🧾 Tickertape View (Embedded)")
    render_tickertape_embed(tickertape_url)
    st.stop()

resolved, data, tried, err = resolve_symbol_and_data(symbol_input, source, alpha_key)

if data is None or resolved is None:
    st.error("Could not resolve symbol / insufficient data from selected source.")
    if err:
        st.warning(f"Provider message: {err}")
    with st.expander("What I tried"):
        st.write(tried)
    st.markdown("### Quick fixes")
    st.write("• Try exact NSE ticker like **RELIANCE.NS**, **TCS.NS**, **INFY.NS**")
    st.write("• For HDFC Bank use **HDFCBANK** (or HDFC)")
    st.write("• Lower **Minimum history required**")
    st.stop()

if len(data) < min_rows:
    st.warning(f"Only {len(data)} daily rows available for {resolved}. Analysis still runs, but confidence is lower.")

st.success(f"Using symbol: {resolved}  |  Rows: {len(data)}")

# Fundamentals
info, fin, bs, cf = ({}, None, None, None)
if source == "Yahoo Finance":
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

if w_ok(20):
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    data["BB_MID"] = bb.bollinger_mavg()
    data["BB_UP"] = bb.bollinger_hband()
    data["BB_LOW"] = bb.bollinger_lband()
else:
    data["BB_MID"] = np.nan
    data["BB_UP"] = np.nan
    data["BB_LOW"] = np.nan

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
    else:
        reasons.append("MACD limited (not enough history)")

    if adx is not None:
        if adx >= 25:
            score += 6; reasons.append(f"ADX strong trend ({adx:.1f})")
        elif adx < 15:
            score -= 4; reasons.append(f"ADX weak trend ({adx:.1f})")
    return int(max(0, min(100, score))), reasons

def fundamental_score():
    reasons = []
    score = 50
    if not info:
        reasons.append("Fundamentals limited (provider didn't return company info).")
        return score, reasons

    def sn(k): return safe_num(info.get(k))
    pe = sn("trailingPE")
    roe = sn("returnOnEquity")
    de = sn("debtToEquity")
    margins = sn("profitMargins")
    rev_g = sn("revenueGrowth")
    earn_g = sn("earningsGrowth")
    fcf = sn("freeCashflow")

    if pe is not None and pe > 0:
        if pe < 18: score += 12; reasons.append(f"Attractive P/E ({pe:.1f})")
        elif pe < 28: score += 6; reasons.append(f"Reasonable P/E ({pe:.1f})")
        else: score -= 6; reasons.append(f"High P/E ({pe:.1f}) — priced for growth")
    else:
        reasons.append("P/E unavailable")

    if roe is not None:
        roe_pct = roe*100
        if roe_pct >= 15: score += 10; reasons.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 8: score += 4; reasons.append(f"Decent ROE ({roe_pct:.1f}%)")
        else: score -= 4; reasons.append(f"Weak ROE ({roe_pct:.1f}%)")

    if margins is not None:
        m = margins*100
        if m >= 12: score += 6; reasons.append(f"Good profit margins ({m:.1f}%)")
        elif m < 5: score -= 4; reasons.append(f"Low margins ({m:.1f}%)")

    if de is not None:
        if de < 80: score += 6; reasons.append(f"Debt/Equity comfortable ({de:.0f})")
        elif de > 180: score -= 8; reasons.append(f"High Debt/Equity ({de:.0f})")

    if rev_g is not None:
        rg = rev_g*100
        if rg >= 12: score += 8; reasons.append(f"Revenue growth strong ({rg:.1f}%)")
        elif rg < 0: score -= 6; reasons.append(f"Revenue shrinking ({rg:.1f}%)")

    if earn_g is not None:
        eg = earn_g*100
        if eg >= 12: score += 8; reasons.append(f"Earnings growth strong ({eg:.1f}%)")
        elif eg < 0: score -= 6; reasons.append(f"Earnings shrinking ({eg:.1f}%)")

    if fcf is not None:
        if fcf > 0: score += 4; reasons.append("Positive free cashflow")
        else: score -= 4; reasons.append("Negative free cashflow")
    return int(max(0, min(100, score))), reasons

t_score, t_reasons = technical_score()
f_score, f_reasons = fundamental_score()

if horizon.startswith("Swing"):
    ai_score = int(round(0.70*t_score + 0.30*f_score))
else:
    ai_score = int(round(0.45*t_score + 0.55*f_score))
label = score_bucket(ai_score)

atr = safe_num(latest.get("ATR14"))
stop = target = None
if atr is not None and price is not None:
    sl_mult, tp_mult = {"Conservative": (1.5, 2.0), "Balanced": (2.0, 3.0), "Aggressive": (2.5, 4.0)}[risk_profile]
    stop = price - sl_mult*atr
    target = price + tp_mult*atr

# UI
top1, top2, top3, top4 = st.columns([1.3, 1, 1, 1])
top1.metric("AI Score (0-100)", f"{ai_score}  {label}")
top2.metric("Technical Score", t_score)
top3.metric("Fundamental Score", f_score)
top4.metric("Last Close", f"{price:.2f}" if price is not None else "N/A")

tabs = st.tabs(["✅ Overview", "📉 Technical", "📘 Fundamentals", "🔍 Troubleshoot"])

with tabs[0]:
    st.subheader("Why this score?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Technical rationale")
        for r in t_reasons[:10]:
            st.write("•", r)
    with c2:
        st.markdown("### Fundamental rationale")
        for r in f_reasons[:12]:
            st.write("•", r)

    st.subheader("Swing planning (optional)")
    c3, c4, c5 = st.columns(3)
    c3.metric("ATR(14)", f"{atr:.2f}" if atr is not None else "N/A")
    c4.metric("Suggested Stop", f"{stop:.2f}" if stop is not None else "N/A")
    c5.metric("Suggested Target", f"{target:.2f}" if target is not None else "N/A")

with tabs[1]:
    st.subheader("Candlestick chart + EMAs")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]
    )])
    for name in ["EMA20", "EMA50", "EMA200"]:
        fig.add_trace(go.Scatter(x=data.index, y=data[name], mode="lines", name=name))
    fig.update_layout(xaxis_rangeslider_visible=False, height=540, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RSI (14)")
        st.line_chart(data["RSI14"])
    with c2:
        st.subheader("MACD")
        st.line_chart(pd.DataFrame({"MACD": data["MACD"], "Signal": data["MACD_SIGNAL"]}))

    st.subheader("Bollinger Bands")
    if "BB_UP" in data.columns:
        st.line_chart(pd.DataFrame({"Close": data["Close"], "BB_UP": data["BB_UP"], "BB_MID": data["BB_MID"], "BB_LOW": data["BB_LOW"]}))

with tabs[2]:
    if not info:
        st.warning("Fundamentals not available from this provider. Use Yahoo Finance source for fundamentals.")
    else:
        st.subheader("Key fundamentals (Yahoo Finance)")
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Market Cap", fmt_big(info.get("marketCap")))
        pe = safe_num(info.get("trailingPE"))
        colB.metric("P/E", f"{pe:.1f}" if pe is not None else "N/A")
        pb = safe_num(info.get("priceToBook"))
        colC.metric("P/B", f"{pb:.2f}" if pb is not None else "N/A")
        dy = safe_num(info.get("dividendYield"))
        colD.metric("Dividend Yield", f"{dy*100:.2f}%" if dy is not None else "N/A")

with tabs[3]:
    st.subheader("Troubleshoot")
    st.write("Symbols tried:")
    st.write(tried)
    st.write("Last provider message:")
    st.write(err or "N/A")

st.divider()
st.caption("Disclaimer: Analytics & scoring tool. Markets involve risk. Verify before investing.")
