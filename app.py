
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import plotly.graph_objects as go
import re
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

@st.cache_data(show_spinner=False, ttl=60*10)
def fetch_price(symbol: str, period: str = "5y", interval: str = "1d"):
    data = yf.download(symbol, period=period, interval=interval, progress=False)
    if data is None or data.empty:
        return None
    data = data.dropna()
    # Ensure OHLC exists
    needed = {"Open","High","Low","Close"}
    if not needed.issubset(set(data.columns)):
        return None
    return data

@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_info(symbol: str):
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        # Statements can fail for some tickers; keep optional
        fin = getattr(t, "financials", None)
        bs = getattr(t, "balance_sheet", None)
        cf = getattr(t, "cashflow", None)
        return info, fin, bs, cf
    except Exception:
        return {}, None, None, None

def try_resolve_symbol(user_symbol: str):
    s = normalize_symbol(user_symbol)
    if not s:
        return None, []
    candidates = [s]
    if "." not in s:
        candidates += [f"{s}.NS", f"{s}.BO"]  # prefer NSE then BSE
    for c in candidates:
        data = fetch_price(c, period="5y", interval="1d")
        if data is not None and len(data) > 250:
            return c, candidates
    return None, candidates

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
    if absn >= 1e12:
        return f"{n/1e12:.2f}T"
    if absn >= 1e9:
        return f"{n/1e9:.2f}B"
    if absn >= 1e6:
        return f"{n/1e6:.2f}M"
    if absn >= 1e3:
        return f"{n/1e3:.2f}K"
    return f"{n:.0f}"

def score_bucket(score):
    if score >= 75:
        return "🟢 Strong"
    if score >= 55:
        return "🟡 Moderate"
    return "🔴 Weak"

# ----------------------------
# Sidebar controls
# ----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    symbol_input = st.text_input("Stock symbol (simple allowed)", "RELIANCE")
    horizon = st.selectbox("Horizon", ["Swing (weeks)", "Long-term (months/years)"])
    risk_profile = st.selectbox("Risk profile", ["Conservative", "Balanced", "Aggressive"])
    show_backtest = st.checkbox("Show quick backtest (demo)", value=True)
    st.divider()
    st.caption("Tip: You can type RELIANCE, TCS, INFY, AAPL. App auto-tries NSE (.NS) and BSE (.BO).")

resolved, tried = try_resolve_symbol(symbol_input)

if not resolved:
    st.error("Could not resolve symbol / insufficient data.")
    with st.expander("Tried these symbols"):
        st.write(tried)
    st.stop()

st.success(f"Using symbol: {resolved}")

# ----------------------------
# Load data
# ----------------------------
data = fetch_price(resolved, period="5y", interval="1d")
if data is None or data.empty:
    st.error("No valid price data found.")
    st.stop()

info, fin, bs, cf = fetch_info(resolved)

close = data["Close"].astype(float)
high = data["High"].astype(float)
low = data["Low"].astype(float)

# ----------------------------
# Technical indicators
# ----------------------------
data["EMA20"] = ta.trend.EMAIndicator(close=close, window=20).ema_indicator()
data["EMA50"] = ta.trend.EMAIndicator(close=close, window=50).ema_indicator()
data["EMA200"] = ta.trend.EMAIndicator(close=close, window=200).ema_indicator()
data["RSI14"] = ta.momentum.RSIIndicator(close=close, window=14).rsi()
macd = ta.trend.MACD(close=close)
data["MACD"] = macd.macd()
data["MACD_SIGNAL"] = macd.macd_signal()
data["ADX14"] = ta.trend.ADXIndicator(high=high, low=low, close=close, window=14).adx()
bb = ta.volatility.BollingerBands(close=close, window=20, window_dev=2)
data["BB_MID"] = bb.bollinger_mavg()
data["BB_UP"] = bb.bollinger_hband()
data["BB_LOW"] = bb.bollinger_lband()
data["ATR14"] = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

latest = data.iloc[-1]
prev = data.iloc[-2] if len(data) > 2 else latest

# ----------------------------
# Technical scoring (0-100)
# ----------------------------
def technical_score():
    reasons = []
    score = 50

    ema20, ema50, ema200 = safe_num(latest["EMA20"]), safe_num(latest["EMA50"]), safe_num(latest["EMA200"])
    rsi = safe_num(latest["RSI14"])
    macd_v, macd_s = safe_num(latest["MACD"]), safe_num(latest["MACD_SIGNAL"])
    adx = safe_num(latest["ADX14"])

    price = safe_num(latest["Close"])

    # Trend
    if all(v is not None for v in [ema20, ema50, ema200, price]):
        if ema20 > ema50 > ema200 and price > ema20:
            score += 18
            reasons.append("Uptrend (EMA20 > EMA50 > EMA200) and price above EMA20")
        elif ema20 < ema50 < ema200 and price < ema20:
            score -= 18
            reasons.append("Downtrend (EMA20 < EMA50 < EMA200) and price below EMA20")
        else:
            reasons.append("Mixed trend (EMAs not aligned)")

    # Momentum (RSI)
    if rsi is not None:
        if 45 <= rsi <= 60:
            score += 6
            reasons.append(f"RSI healthy ({rsi:.1f})")
        elif rsi < 35:
            score += 4
            reasons.append(f"RSI oversold zone ({rsi:.1f}) — potential bounce")
        elif rsi > 70:
            score -= 6
            reasons.append(f"RSI overbought ({rsi:.1f}) — caution")

    # MACD cross
    if macd_v is not None and macd_s is not None:
        if macd_v > macd_s:
            score += 8
            reasons.append("MACD above signal (bullish momentum)")
        else:
            score -= 6
            reasons.append("MACD below signal (bearish momentum)")

    # Trend strength (ADX)
    if adx is not None:
        if adx >= 25:
            score += 6
            reasons.append(f"ADX strong trend ({adx:.1f})")
        elif adx < 15:
            score -= 4
            reasons.append(f"ADX weak trend ({adx:.1f})")

    return int(max(0, min(100, score))), reasons

t_score, t_reasons = technical_score()

# ----------------------------
# Fundamental scoring (0-100) from yfinance info
# ----------------------------
def fundamental_score():
    reasons = []
    score = 50

    pe = safe_num(info.get("trailingPE"))
    pb = safe_num(info.get("priceToBook"))
    roe = safe_num(info.get("returnOnEquity"))
    roa = safe_num(info.get("returnOnAssets"))
    de = safe_num(info.get("debtToEquity"))
    margins = safe_num(info.get("profitMargins"))
    rev_g = safe_num(info.get("revenueGrowth"))
    earn_g = safe_num(info.get("earningsGrowth"))
    fcf = safe_num(info.get("freeCashflow"))

    # Valuation (PE)
    if pe is not None and pe > 0:
        if pe < 18:
            score += 12; reasons.append(f"Attractive P/E ({pe:.1f})")
        elif pe < 28:
            score += 6; reasons.append(f"Reasonable P/E ({pe:.1f})")
        else:
            score -= 6; reasons.append(f"High P/E ({pe:.1f}) — priced for growth")
    else:
        reasons.append("P/E unavailable")

    # Profitability
    if roe is not None:
        roe_pct = roe * 100
        if roe_pct >= 15:
            score += 10; reasons.append(f"Strong ROE ({roe_pct:.1f}%)")
        elif roe_pct >= 8:
            score += 4; reasons.append(f"Decent ROE ({roe_pct:.1f}%)")
        else:
            score -= 4; reasons.append(f"Weak ROE ({roe_pct:.1f}%)")
    if margins is not None:
        m_pct = margins * 100
        if m_pct >= 12:
            score += 6; reasons.append(f"Good profit margins ({m_pct:.1f}%)")
        elif m_pct < 5:
            score -= 4; reasons.append(f"Low margins ({m_pct:.1f}%)")

    # Leverage
    if de is not None:
        if de < 80:
            score += 6; reasons.append(f"Debt/Equity comfortable ({de:.0f})")
        elif de > 180:
            score -= 8; reasons.append(f"High Debt/Equity ({de:.0f})")
    # Growth
    if rev_g is not None:
        rg = rev_g * 100
        if rg >= 12:
            score += 8; reasons.append(f"Revenue growth strong ({rg:.1f}%)")
        elif rg < 0:
            score -= 6; reasons.append(f"Revenue shrinking ({rg:.1f}%)")
    if earn_g is not None:
        eg = earn_g * 100
        if eg >= 12:
            score += 8; reasons.append(f"Earnings growth strong ({eg:.1f}%)")
        elif eg < 0:
            score -= 6; reasons.append(f"Earnings shrinking ({eg:.1f}%)")

    # Free cash flow (stability)
    if fcf is not None:
        if fcf > 0:
            score += 4; reasons.append("Positive free cashflow")
        else:
            score -= 4; reasons.append("Negative free cashflow")

    return int(max(0, min(100, score))), reasons

f_score, f_reasons = fundamental_score()

# ----------------------------
# Composite AI Score
# ----------------------------
if horizon.startswith("Swing"):
    ai_score = int(round(0.70 * t_score + 0.30 * f_score))
else:
    ai_score = int(round(0.45 * t_score + 0.55 * f_score))

label = score_bucket(ai_score)

# Risk controls / swing levels (ATR-based)
atr = safe_num(latest["ATR14"])
price = safe_num(latest["Close"])
stop = None
target = None
if atr is not None and price is not None:
    if risk_profile == "Conservative":
        stop = price - 1.5 * atr
        target = price + 2.0 * atr
    elif risk_profile == "Balanced":
        stop = price - 2.0 * atr
        target = price + 3.0 * atr
    else:
        stop = price - 2.5 * atr
        target = price + 4.0 * atr

# ----------------------------
# Layout
# ----------------------------
top1, top2, top3, top4 = st.columns([1.2, 1, 1, 1])
top1.metric("AI Score (0-100)", f"{ai_score}  {label}")
top2.metric("Technical Score", t_score)
top3.metric("Fundamental Score", f_score)
top4.metric("Last Close", f"{price:.2f}" if price is not None else "N/A")

st.caption("Higher score = stronger overall setup, based on rule-based scoring. Always verify with your own judgement.")

tab_over, tab_tech, tab_fund, tab_back = st.tabs(["✅ Overview", "📉 Technical", "📘 Fundamentals", "🧪 Backtest"])

with tab_over:
    st.subheader("Why this score?")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Technical rationale")
        for r in t_reasons[:8]:
            st.write("•", r)
    with c2:
        st.markdown("### Fundamental rationale")
        for r in f_reasons[:10]:
            st.write("•", r)

    st.subheader("Swing trade planning (optional)")
    c3, c4, c5 = st.columns(3)
    c3.metric("ATR(14)", f"{atr:.2f}" if atr is not None else "N/A")
    c4.metric("Suggested Stop", f"{stop:.2f}" if stop is not None else "N/A")
    c5.metric("Suggested Target", f"{target:.2f}" if target is not None else "N/A")
    st.caption("Stop/Target are ATR-based guides, not guarantees.")

with tab_tech:
    st.subheader("Candlestick chart")
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    )])
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA20"], mode="lines", name="EMA20"))
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA50"], mode="lines", name="EMA50"))
    fig.add_trace(go.Scatter(x=data.index, y=data["EMA200"], mode="lines", name="EMA200"))
    fig.update_layout(xaxis_rangeslider_visible=False, height=520, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RSI (14)")
        st.line_chart(data["RSI14"])
    with c2:
        st.subheader("MACD")
        macd_df = pd.DataFrame({"MACD": data["MACD"], "Signal": data["MACD_SIGNAL"]})
        st.line_chart(macd_df)

    st.subheader("Bollinger Bands")
    bb_df = pd.DataFrame({"Close": data["Close"], "BB_UP": data["BB_UP"], "BB_MID": data["BB_MID"], "BB_LOW": data["BB_LOW"]})
    st.line_chart(bb_df)

with tab_fund:
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

with tab_back:
    if not show_backtest:
        st.info("Enable backtest from sidebar to view.")
    else:
        st.subheader("Quick backtest (demo) — Trend + RSI filter")
        st.caption("This is a simple historical simulation to sanity-check the rules. Not optimized and not a guarantee.")

        bt = data.copy().dropna()
        bt["signal"] = 0

        # Entry: uptrend + RSI bounce
        bt.loc[(bt["EMA20"] > bt["EMA50"]) & (bt["Close"] > bt["EMA200"]) & (bt["RSI14"] < 45), "signal"] = 1
        # Exit: RSI overbought or trend break
        bt.loc[(bt["RSI14"] > 65) | (bt["EMA20"] < bt["EMA50"]), "signal"] = 0

        bt["position"] = bt["signal"].replace(to_replace=0, method="ffill").fillna(0)
        bt["ret"] = bt["Close"].pct_change().fillna(0.0)
        bt["strategy_ret"] = bt["position"].shift(1).fillna(0) * bt["ret"]
        bt["equity"] = (1 + bt["strategy_ret"]).cumprod()
        bt["buyhold"] = (1 + bt["ret"]).cumprod()

        # Performance metrics
        total = bt["equity"].iloc[-1] - 1
        bh_total = bt["buyhold"].iloc[-1] - 1

        # Max drawdown
        roll_max = bt["equity"].cummax()
        dd = (bt["equity"] / roll_max) - 1
        max_dd = dd.min()

        c1, c2, c3 = st.columns(3)
        c1.metric("Strategy Return", f"{total*100:.1f}%")
        c2.metric("Buy & Hold Return", f"{bh_total*100:.1f}%")
        c3.metric("Max Drawdown", f"{max_dd*100:.1f}%")

        st.line_chart(pd.DataFrame({"Strategy": bt["equity"], "Buy&Hold": bt["buyhold"]}))

        with st.expander("Backtest data"):
            st.dataframe(bt[["Close","EMA20","EMA50","EMA200","RSI14","position","equity","buyhold"]].tail(200))

# Footer
st.divider()
st.caption("Disclaimer: This tool provides analytics and rule-based scoring. Markets involve risk. Use position sizing and verify with multiple sources before investing.")
