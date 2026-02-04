
import streamlit as st
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import yfinance as yf
import re
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Investment Platform (NSE/BSE Ready)", layout="wide")
st.title("🧠 AI Investment Platform (Swing + Long-Term)")
st.caption("Decision-support only (not financial advice).")

# Google Finance & Tickertape embeds failed because those sites block iframe embedding (403 / blank) using security headers.
# That's not a code bug; it’s their policy. So we replace them with a TradingView widget that works inside Streamlit,
# and we offer direct NSE/BSE data options via broker APIs (Zerodha/Upstox/Angel), which are stable and legal.

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

def score_bucket(score):
    if score >= 75: return "🟢 Strong"
    if score >= 55: return "🟡 Moderate"
    return "🔴 Weak"

INDIA_ALIAS = {
    "HDFC": "HDFCBANK",
    "ICICI": "ICICIBANK",
    "KOTAK": "KOTAKBANK",
    "BAJAJFIN": "BAJFINANCE",
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
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False, threads=False)
        df = _ensure_ohlc(df)
        if df is not None and not df.empty:
            return df, None
    except Exception as e:
        err1 = str(e)
    else:
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

def resolve_data(user_symbol: str, source: str):
    candidates = build_candidates(user_symbol)
    last_err = None

    if source != "Yahoo Finance (Free fallback)":
        # Broker data not configured in this deployable template.
        return None, None, candidates, f"{source} selected but not configured. Add API keys/tokens in Streamlit Secrets."

    for c in candidates:
        df, err = fetch_price_yahoo(c, period="5y")
        if df is not None and len(df) >= 30:
            return c, df, candidates, None
        last_err = err

    return None, None, candidates, last_err

def tradingview_symbol(resolved_symbol: str) -> str:
    rs = resolved_symbol.upper()
    if rs.endswith(".NS"):
        return f"NSE:{rs[:-3]}"
    if rs.endswith(".BO"):
        return f"BSE:{rs[:-3]}"
    return rs

def render_tradingview(symbol_tv: str):
    html = f"""
    <div class="tradingview-widget-container">
      <div id="tradingview_1" style="height: 620px;"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{symbol_tv}",
        "interval": "D",
        "timezone": "Asia/Kolkata",
        "theme": "light",
        "style": "1",
        "locale": "en",
        "toolbar_bg": "#f1f3f6",
        "enable_publishing": false,
        "withdateranges": true,
        "hide_side_toolbar": false,
        "allow_symbol_change": true,
        "details": true,
        "hotlist": false,
        "calendar": false,
        "container_id": "tradingview_1"
      }});
      </script>
    </div>
    """
    components.html(html, height=650)

with st.sidebar:
    st.header("⚙️ Settings")
    symbol_input = st.text_input("Script / Symbol (simple OK)", "RELIANCE")
    data_source = st.selectbox(
        "Data source",
        [
            "Yahoo Finance (Free fallback)",
            "Zerodha Kite (NSE/BSE) — connect API",
            "Upstox (NSE/BSE) — connect API",
            "Angel One SmartAPI (NSE/BSE) — connect API",
        ],
    )
    min_rows = st.slider("Minimum history required (days)", 30, 400, 90)
    horizon = st.selectbox("Horizon", ["Swing (weeks)", "Long-term (months/years)"])
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"])
    show_tradingview = st.checkbox("Show TradingView research panel (inside app)", value=True)

run = st.button("✅ Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter symbol and click **Run Analysis**.")
    st.stop()

resolved, data, tried, err = resolve_data(symbol_input, data_source)

if data is None or resolved is None:
    st.error("Could not resolve symbol / insufficient data from selected source.")
    if err:
        st.warning(f"Provider message: {err}")
    with st.expander("Tried symbols"):
        st.write(tried)
    st.markdown("### Fix now")
    st.write("• Try exact NSE: **RELIANCE.NS**, **TCS.NS**, **INFY.NS**")
    st.write("• Try exact BSE: **RELIANCE.BO**")
    st.write("• For direct NSE/BSE source: select your broker option (Zerodha/Upstox/Angel) and connect API keys (see Setup tab).")
    st.stop()

if len(data) < min_rows:
    st.warning(f"Only {len(data)} rows for {resolved}. Analysis will run but confidence is lower.")

st.success(f"Using: {resolved}  |  Rows: {len(data)}")

if show_tradingview:
    st.subheader("📌 Research Panel (inside app)")
    render_tradingview(tradingview_symbol(resolved))
    st.divider()

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
        reasons.append("Trend limited (not enough history)")

    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 6; reasons.append(f"RSI healthy ({rsi:.1f})")
        elif rsi < 35:
            score += 4; reasons.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            score -= 6; reasons.append(f"RSI overbought ({rsi:.1f})")

    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            score += 8; reasons.append("MACD bullish")
        else:
            score -= 6; reasons.append("MACD bearish")

    if adx is not None:
        if adx >= 25:
            score += 6; reasons.append(f"ADX strong trend ({adx:.1f})")
        elif adx < 15:
            score -= 4; reasons.append(f"ADX weak trend ({adx:.1f})")

    return int(max(0, min(100, score))), reasons

t_score, t_reasons = technical_score()
f_score = 50  # placeholder unless you connect a fundamentals provider
ai_score = int(round((0.75*t_score + 0.25*f_score) if horizon.startswith("Swing") else (0.55*t_score + 0.45*f_score)))

atr = safe_num(latest.get("ATR14"))
stop = target = None
if atr is not None and price is not None:
    sl_mult, tp_mult = {"Conservative": (1.5, 2.0), "Balanced": (2.0, 3.0), "Aggressive": (2.5, 4.0)}[risk_profile]
    stop = price - sl_mult*atr
    target = price + tp_mult*atr

c1, c2, c3, c4 = st.columns([1.4, 1, 1, 1])
c1.metric("AI Score (0-100)", f"{ai_score}  {score_bucket(ai_score)}")
c2.metric("Technical Score", t_score)
c3.metric("Suggested Stop", f"{stop:.2f}" if stop is not None else "N/A")
c4.metric("Suggested Target", f"{target:.2f}" if target is not None else "N/A")

tabs = st.tabs(["📉 Chart", "🧠 Reasons", "🔧 Setup Direct NSE/BSE"])

with tabs[0]:
    st.subheader("Candlestick chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index, open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"]
    )])
    for name in ["EMA20", "EMA50", "EMA200"]:
        fig.add_trace(go.Scatter(x=data.index, y=data[name], mode="lines", name=name))
    fig.update_layout(xaxis_rangeslider_visible=False, height=560)
    st.plotly_chart(fig, use_container_width=True)

with tabs[1]:
    st.subheader("Why this signal?")
    for r in t_reasons:
        st.write("•", r)

with tabs[2]:
    st.subheader("Direct NSE/BSE data (recommended)")
    st.write("For reliable direct NSE/BSE candles, use a broker API (official & stable).")
    st.markdown("""
**How to enable (simple):**
1. Select your broker in the sidebar (Zerodha / Upstox / Angel One)  
2. Add API keys/tokens in Streamlit Cloud → **App settings → Secrets**  
3. We update the app to fetch candles from broker endpoints

**What I need from you:** which broker you use (Zerodha / Upstox / Angel / Dhan / Fyers)  
Then I will plug the exact working integration code for that broker.
""")
    st.info("Google Finance / Tickertape cannot be embedded due to their security policy. TradingView works inside app.")

st.divider()
st.caption("Disclaimer: Analytics tool. Markets involve risk.")
