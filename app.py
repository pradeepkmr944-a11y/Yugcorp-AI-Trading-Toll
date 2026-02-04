
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import re
import requests
from datetime import datetime

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

# A small India-friendly mapper for very common shorthand names.
# You can extend this list anytime.
INDIA_ALIAS = {
    "HDFC": "HDFCBANK",
    "ICICI": "ICICIBANK",
    "BAJAJFIN": "BAJFINANCE",
    "MOTHERSUMI": "MSUMI",
    "LTI": "LTIM",
    "M&M": "M&M",
    "MM": "M&M",
    "MARUTI": "MARUTI",
}

# ----------------------------
# Data fetchers
# ----------------------------
@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_price_yahoo(symbol: str, period: str = "5y", interval: str = "1d"):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data is None or data.empty:
        return None
    data = data.dropna()
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(set(data.columns)):
        return None
    return data

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

def _alpha_to_ohlc(ts_json: dict):
    # Alpha Vantage daily series: "Time Series (Daily)"
    key = None
    for k in ts_json.keys():
        if "Time Series" in k:
            key = k
            break
    if not key:
        return None
    rows = []
    for dt, vals in ts_json[key].items():
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
    df = pd.DataFrame(rows).sort_values("Date").set_index("Date")
    return df

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_price_alpha_vantage(symbol: str, api_key: str):
    # Works best for US/global tickers; NSE requires special symbols and may be inconsistent.
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "full"
    }
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
    df = df.dropna()
    return df, None

def build_candidates(user_symbol: str):
    s = normalize_symbol(user_symbol)
    if not s:
        return []
    # apply alias mapping if exists
    base = INDIA_ALIAS.get(s, s)

    candidates = []
    # if user already added suffix like .NS, use as-is first
    candidates.append(base)
    if "." not in base:
        # prefer NSE then BSE
        candidates.extend([f"{base}.NS", f"{base}.BO"])
    return candidates

def resolve_symbol_and_data(user_symbol: str, source: str, alpha_key: str | None):
    candidates = build_candidates(user_symbol)
    last_error = None

    for c in candidates:
        if source == "Yahoo Finance":
            d = fetch_price_yahoo(c, period="5y", interval="1d")
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
        else:
            if not alpha_key:
                last_error = "Alpha Vantage API key missing"
                continue
            d, err = fetch_price_alpha_vantage(c, alpha_key)
            if d is not None and len(d) >= 30:
                return c, d, candidates, None
            last_error = err

    return None, None, candidates, last_error

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol_input = st.text_input("Stock symbol / script name", "RELIANCE")
    source = st.selectbox("Data source", ["Yahoo Finance", "Alpha Vantage (needs free API key)"])
    alpha_key = None
    if source.startswith("Alpha"):
        alpha_key = st.text_input("Alpha Vantage API Key (free)", type="password")
        st.caption("Get a free key from Alpha Vantage website. If you trade NSE symbols, Yahoo Finance usually works better.")
    horizon = st.selectbox("Horizon", ["Swing (weeks)", "Long-term (months/years)"])
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"])
    min_rows = st.slider("Minimum history required (days)", 30, 400, 90)
    show_backtest = st.checkbox("Show quick backtest (demo)", value=True)
    st.divider()
    st.caption("You can type: RELIANCE, TCS, INFY, HDFC (auto-maps to HDFCBANK), AAPL, MSFT, TSLA")

resolved, data, tried, err = resolve_symbol_and_data(symbol_input, source, alpha_key)

if not symbol_input.strip():
    st.info("Please type a stock symbol (e.g., RELIANCE or AAPL).")
    st.stop()

if data is None or resolved is None:
    st.error("Could not resolve symbol / insufficient data from selected source.")
    if err:
        st.warning(f"Provider message: {err}")
    with st.expander("What I tried"):
        st.write(tried)
    st.info("Quick fixes: (1) Try exact symbol like RELIANCE.NS, TCS.NS. (2) Increase/Decrease minimum history. (3) Switch source.")
    st.stop()

if len(data) < min_rows:
    st.warning(f"Only {len(data)} daily rows available for {resolved}. Analysis will still run, but confidence is reduced. Try another symbol or lower minimum history in sidebar.")

st.success(f"Using symbol: {resolved}  |  Rows: {len(data)}")

# Info / statements (Yahoo only reliably provides fundamentals)
info, fin, bs, cf = ({}, None, None, None)
if source == "Yahoo Finance":
    info, fin, bs, cf = fetch_info_yahoo(resolved)

# ----------------------------
# Indicators
# ----------------------------
data = data.copy().dropna()
close = data["Close"].astype(float)
high = data["High"].astype(float)
low = data["Low"].astype(float)

# If history is short, guard windows
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

if w_ok(20):
    bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
    data["BB_MID"] = bb.bollinger_mavg()
    data["BB_UP"] = bb.bollinger_hband()
    data["BB_LOW"] = bb.bollinger_lband()
else:
    data["BB_MID"] = np.nan
    data["BB_UP"] = np.nan
    data["BB_LOW"] = np.nan

data["ATR14"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range() if w_ok(14) else np.nan

latest = data.iloc[-1]

# ----------------------------
# Scoring
# ----------------------------
def technical_score():
    reasons = []
    score = 50

    ema20, ema50, ema200 = safe_num(latest.get("EMA20")), safe_num(latest.get("EMA50")), safe_num(latest.get("EMA200"))
    rsi = safe_num(latest.get("RSI14"))
    macd_v, macd_s = safe_num(latest.get("MACD")), safe_num(latest.get("MACD_SIGNAL"))
    adx = safe_num(latest.get("ADX14"))
    price = safe_num(latest.get("Close"))

    # Trend
    if all(v is not None for v in [ema20, ema50, ema200, price]):
        if ema20 > ema50 > ema200 and price > ema20:
            score += 18; reasons.append("Uptrend (EMA20 > EMA50 > EMA200) and price above EMA20")
        elif ema20 < ema50 < ema200 and price < ema20:
            score -= 18; reasons.append("Downtrend (EMA20 < EMA50 < EMA200) and price below EMA20")
        else:
            reasons.append("Mixed trend (EMAs not aligned)")
    else:
        reasons.append("Trend score limited (not enough history for EMAs)")

    # RSI
    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 6; reasons.append(f"RSI healthy ({rsi:.1f})")
        elif rsi < 35:
            score += 4; reasons.append(f"RSI oversold ({rsi:.1f}) — potential bounce")
        elif rsi > 70:
            score -= 6; reasons.append(f"RSI overbought ({rsi:.1f}) — caution")

    # MACD
    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            score += 8; reasons.append("MACD above signal (bullish momentum)")
        else:
            score -= 6; reasons.append("MACD below signal (bearish momentum)")
    else:
        reasons.append("MACD limited (not enough history)")

    # ADX
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

    pe = safe_num(info.get("trailingPE"))
    roe = safe_num(info.get("returnOnEquity"))
    de = safe_num(info.get("debtToEquity"))
    margins = safe_num(info.get("profitMargins"))
    rev_g = safe_num(info.get("revenueGrowth"))
    earn_g = safe_num(info.get("earningsGrowth"))
    fcf = safe_num(info.get("freeCashflow"))

    # PE
    if pe is not None and pe > 0:
        if pe < 18: score += 12; reasons.append(f"Attractive P/E ({pe:.1f})")
        elif pe < 28: score += 6; reasons.append(f"Reasonable P/E ({pe:.1f})")
        else: score -= 6; reasons.append(f"High P/E ({pe:.1f}) — priced for growth")
    else:
        reasons.append("P/E unavailable")

    # ROE
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 15: score += 10; reasons.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 8: score += 4; reasons.append(f"Decent ROE ({roe_pct:.1f}%)")
        else: score -= 4; reasons.append(f"Weak ROE ({roe_pct:.1f}%)")

    # Margins
    if margins is not None:
        m = margins * 100
        if m >= 12: score += 6; reasons.append(f"Good profit margins ({m:.1f}%)")
        elif m < 5: score -= 4; reasons.append(f"Low margins ({m:.1f}%)")

    # Leverage
    if de is not None:
        if de < 80: score += 6; reasons.append(f"Debt/Equity comfortable ({de:.0f})")
        elif de > 180: score -= 8; reasons.append(f"High Debt/Equity ({de:.0f})")

    # Growth
    if rev_g is not None:
        rg = rev_g * 100
        if rg >= 12: score += 8; reasons.append(f"Revenue growth strong ({rg:.1f}%)")
        elif rg < 0: score -= 6; reasons.append(f"Revenue shrinking ({rg:.1f}%)")
    if earn_g is not None:
        eg = earn_g * 100
        if eg >= 12: score += 8; reasons.append(f"Earnings growth strong ({eg:.1f}%)")
        elif eg < 0: score -= 6; reasons.append(f"Earnings shrinking ({eg:.1f}%)")

    # FCF
    if fcf is not None:
        if fcf > 0: score += 4; reasons.append("Positive free cashflow")
        else: score -= 4; reasons.append("Negative free cashflow")

    return int(max(0, min(100, score))), reasons

t_score, t_reasons = technical_score()
f_score, f_reasons = fundamental_score()

if horizon.startswith("Swing"):
    ai_score = int(round(0.70 * t_score + 0.30 * f_score))
else:
    ai_score = int(round(0.45 * t_score + 0.55 * f_score))

label = score_bucket(ai_score)

# ATR stop/target
atr = safe_num(latest.get("ATR14"))
price = safe_num(latest.get("Close"))
stop = target = None
if atr is not None and price is not None:
    mult = {"Conservative": (1.5, 2.0), "Balanced": (2.0, 3.0), "Aggressive": (2.5, 4.0)}[risk_profile]
    stop = price - mult[0]*atr
    target = price + mult[1]*atr

# ----------------------------
# UI
# ----------------------------
top1, top2, top3, top4 = st.columns([1.3, 1, 1, 1])
top1.metric("AI Score (0-100)", f"{ai_score}  {label}")
top2.metric("Technical Score", t_score)
top3.metric("Fundamental Score", f_score)
top4.metric("Last Close", f"{price:.2f}" if price is not None else "N/A")

tabs = st.tabs(["✅ Overview", "📉 Technical", "📘 Fundamentals", "🧪 Backtest", "🔍 Troubleshoot"])

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
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )])
    for name in ["EMA20", "EMA50", "EMA200"]:
        if name in data.columns:
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
    st.line_chart(pd.DataFrame({"Close": data["Close"], "BB_UP": data["BB_UP"], "BB_MID": data["BB_MID"], "BB_LOW": data["BB_LOW"]}))

with tabs[2]:
    if not info:
        st.warning("Fundamentals not available from selected provider. Switch data source to Yahoo Finance for fundamentals.")
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

        colE, colF, colG, colH = st.columns(4)
        roe = safe_num(info.get("returnOnEquity"))
        colE.metric("ROE", f"{roe*100:.1f}%" if roe is not None else "N/A")
        pm = safe_num(info.get("profitMargins"))
        colF.metric("Profit Margin", f"{pm*100:.1f}%" if pm is not None else "N/A")
        de = safe_num(info.get("debtToEquity"))
        colG.metric("Debt/Equity", f"{de:.0f}" if de is not None else "N/A")
        rg = safe_num(info.get("revenueGrowth"))
        colH.metric("Revenue Growth", f"{rg*100:.1f}%" if rg is not None else "N/A")

        with st.expander("Company profile"):
            st.write({
                "Name": info.get("shortName"),
                "Sector": info.get("sector"),
                "Industry": info.get("industry"),
                "Website": info.get("website"),
                "Summary": info.get("longBusinessSummary")
            })

        with st.expander("Financial statements (if available)"):
            if fin is not None and isinstance(fin, pd.DataFrame) and not fin.empty:
                st.markdown("**Income Statement (annual)**")
                st.dataframe(fin)
            else:
                st.info("Income statement not available for this symbol via Yahoo Finance.")

            if bs is not None and isinstance(bs, pd.DataFrame) and not bs.empty:
                st.markdown("**Balance Sheet (annual)**")
                st.dataframe(bs)
            else:
                st.info("Balance sheet not available for this symbol via Yahoo Finance.")

            if cf is not None and isinstance(cf, pd.DataFrame) and not cf.empty:
                st.markdown("**Cashflow (annual)**")
                st.dataframe(cf)
            else:
                st.info("Cashflow not available for this symbol via Yahoo Finance.")

with tabs[3]:
    st.subheader("Quick backtest (demo) — Trend + RSI filter")
    st.caption("Simple simulation to sanity-check rules. Not optimized; not a guarantee.")
    bt = data.copy().dropna()
    bt["signal"] = 0
    # Entry: EMA20>EMA50 and RSI < 45
    bt.loc[(bt["EMA20"] > bt["EMA50"]) & (bt["RSI14"] < 45), "signal"] = 1
    # Exit: RSI > 65 or EMA20 < EMA50
    bt.loc[(bt["RSI14"] > 65) | (bt["EMA20"] < bt["EMA50"]), "signal"] = 0
    bt["position"] = bt["signal"].replace(to_replace=0, method="ffill").fillna(0)
    bt["ret"] = bt["Close"].pct_change().fillna(0.0)
    bt["strategy_ret"] = bt["position"].shift(1).fillna(0) * bt["ret"]
    bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
    bt["buyhold"] = (1 + bt["ret"]).cumprod()
    total = bt["equity"].iloc[-1] - 1
    bh_total = bt["buyhold"].iloc[-1] - 1
    roll_max = bt["equity"].cummax()
    dd = (bt["equity"] / roll_max) - 1
    max_dd = dd.min()
    c1, c2, c3 = st.columns(3)
    c1.metric("Strategy Return", f"{total*100:.1f}%")
    c2.metric("Buy & Hold Return", f"{bh_total*100:.1f}%")
    c3.metric("Max Drawdown", f"{max_dd*100:.1f}%")
    st.line_chart(pd.DataFrame({"Strategy": bt["equity"], "Buy&Hold": bt["buyhold"]}))
    with st.expander("Backtest data"):
        st.dataframe(bt.tail(200))

with tabs[4]:
    st.subheader("Troubleshooting")
    st.write("If you see **Could not resolve symbol**, try:")
    st.write("1) Use the exact NSE symbol: **RELIANCE.NS**, **TCS.NS**, **INFY.NS**")
    st.write("2) For HDFC Bank, type **HDFCBANK** (app also maps HDFC → HDFCBANK)")
    st.write("3) Lower **Minimum history required** in the sidebar")
    st.write("4) Switch data source to Alpha Vantage (and enter a free API key) for US stocks")
    st.write("5) Yahoo Finance sometimes has symbol differences; search the correct ticker on Yahoo Finance and copy it here.")
    with st.expander("Symbols tried"):
        st.write(tried)

st.divider()
st.caption("Disclaimer: Analytics & scoring tool. Markets involve risk. Verify from official filings and multiple sources before investing.")
