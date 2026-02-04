
import streamlit as st
import yfinance as yf
import ta

st.set_page_config(page_title="AI Trading Bot", layout="wide")

st.title("📈 AI Stock Trading Signal App")

symbol = st.text_input("Enter Stock Symbol (NSE Format)", "RELIANCE.NS")

if st.button("Run AI Scan"):
    data = yf.download(symbol, period="5d", interval="5m")

    if data.empty:
        st.error("No data found. Check symbol.")
    else:
        data["rsi"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
        latest = data.iloc[-1]

        if latest["rsi"] < 30:
            signal = "BUY"
        elif latest["rsi"] > 70:
            signal = "SELL"
        else:
            signal = "HOLD"

        st.metric("AI Signal", signal)
        st.line_chart(data["Close"])
