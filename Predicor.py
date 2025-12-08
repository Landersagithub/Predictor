import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from twelvedata import TDClient
import plotly.graph_objects as go

# ------------------ CONFIG ------------------
st.set_page_config(page_title="Stock Price App", layout="wide")
TD_API_KEY = "27db7c0275cc41a0a9d174497a08c337"  # Replace with your Twelve Data API key
td = TDClient(apikey=TD_API_KEY)
USD_TO_PHP = 56.50

# ------------------ SESSION STATES ------------------
for key in ['show_latest','show_history','show_chart','show_predict','show_buy','show_sell']:
    if key not in st.session_state:
        st.session_state[key] = False

# ------------------ APP HEADER ------------------
st.title("Stock Price Analysis Application")
st.info(f"🌍 International Stocks: Twelve Data | 💱 Live Exchange Rate: 1 USD = ₱{USD_TO_PHP:.2f}")
st.caption("**International stocks:** AAPL, TSLA, MSFT | **Philippine stocks:** JFC, SM, BDO, ALI (auto .PSE suffix)")

symbol = st.text_input("Enter stock symbol:")

# ------------------ ACTION BUTTONS ------------------
col1, col2, col3 = st.columns(3)
if col1.button("Get Latest Price"): st.session_state.update({k: k=='show_latest' for k in st.session_state})
if col2.button("Show Historical Data"): st.session_state.update({k: k=='show_history' for k in st.session_state})
if col3.button("Show Chart"): st.session_state.update({k: k=='show_chart' for k in st.session_state})

col4, col5, col6 = st.columns(3)
if col4.button("Predict Future Prices"): st.session_state.update({k: k=='show_predict' for k in st.session_state})
if col5.button("Buy Recommendation"): st.session_state.update({k: k=='show_buy' for k in st.session_state})
if col6.button("Sell Analysis"): st.session_state.update({k: k=='show_sell' for k in st.session_state})

# ------------------ HELPER FUNCTIONS ------------------
def is_pse_stock(symbol):
    pse_list = ['JFC','SM','BDO','ALI','MBT','BPI','SMPH','MEG','TEL','GLO','AC','AGI','MER','DMC','URC',
                'PGOLD','LTG','ICT','RLC','AEV','COL','BLOOM','CNPF','CEB','SMDC','FGEN','AP','EMI','HOUSE','CLI']
    return symbol.upper() in pse_list

@st.cache_data(ttl=3600)
def fetch_twelvedata(symbol, interval="1day", outputsize=365):
    try:
        sym = symbol.upper()
        if is_pse_stock(sym) and not sym.endswith(".PSE"):
            sym += ".PSE"
        ts = td.time_series(symbol=sym, interval=interval, outputsize=outputsize, timezone="Asia/Manila")
        data = ts.as_pandas()
        if data.empty: return None
        data.index = pd.to_datetime(data.index)
        data = data.sort_index(ascending=False)
        data.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
        return data
    except Exception as e:
        st.error(f"Twelve Data fetch error: {e}")
        return None

# ------------------ MAIN ------------------
if symbol:
    with st.spinner(f"Fetching data for {symbol.upper()}..."):
        hist_df = fetch_twelvedata(symbol)
        if hist_df is None:
            st.error(f"Unable to fetch data for: {symbol.upper()}")
        else:
            is_pse = is_pse_stock(symbol)
            hist_df['Open_PHP'] = hist_df['Open'] if is_pse else hist_df['Open']*USD_TO_PHP
            hist_df['High_PHP'] = hist_df['High'] if is_pse else hist_df['High']*USD_TO_PHP
            hist_df['Low_PHP'] = hist_df['Low'] if is_pse else hist_df['Low']*USD_TO_PHP
            hist_df['Close_PHP'] = hist_df['Close'] if is_pse else hist_df['Close']*USD_TO_PHP

            hist_df = hist_df[['Open','High','Low','Close','Volume','Open_PHP','High_PHP','Low_PHP','Close_PHP']]
            hist_df.sort_index(ascending=False, inplace=True)

            # ---------- LATEST PRICE ----------
            if st.session_state.show_latest:
                st.subheader("Latest Stock Price")
                latest_row = hist_df.iloc[0]
                prev_close = hist_df.iloc[1]['Close_PHP'] if len(hist_df)>1 else latest_row['Close_PHP']
                change = latest_row['Close_PHP'] - prev_close
                change_pct = (change / prev_close) * 100
                change_color = "green" if change>=0 else "red"
                st.markdown(f"""
                    <div style='padding:20px; border:1px solid #ddd; border-radius:10px;'>
                        <h3>{symbol.upper()}</h3>
                        <p style='font-size:24px; font-weight:bold; color:{change_color};'>
                        ₱{latest_row['Close_PHP']:.2f} ({change:+.2f}, {change_pct:+.2f}%)</p>
                    </div>
                """, unsafe_allow_html=True)

            # ---------- HISTORICAL DATA ----------
            if st.session_state.show_history:
                st.subheader("Historical Data")
                st.dataframe(hist_df[['Open_PHP','High_PHP','Low_PHP','Close_PHP','Volume']].style.format({
                    "Open_PHP":"₱{:.2f}","High_PHP":"₱{:.2f}","Low_PHP":"₱{:.2f}","Close_PHP":"₱{:.2f}","Volume":"{:,}"
                }), use_container_width=True)
                csv = hist_df.to_csv()
                st.download_button("Download CSV", csv, f"{symbol}_data.csv")

            # ---------- CHART ----------
            if st.session_state.show_chart:
                st.subheader("Stock Charts")
                df_chart = hist_df.sort_index(ascending=True)
                st.line_chart(df_chart['Close_PHP'])
                candle_fig = go.Figure(data=[go.Candlestick(
                    x=df_chart.index,
                    open=df_chart['Open_PHP'], high=df_chart['High_PHP'],
                    low=df_chart['Low_PHP'], close=df_chart['Close_PHP'],
                    name=symbol.upper()
                )])
                st.plotly_chart(candle_fig, use_container_width=True)

            # ---------- PREDICTION ----------
            if st.session_state.show_predict:
                if len(hist_df)<30:
                    st.error("Not enough data for prediction (minimum 30 days)")
                else:
                    st.subheader("Future Price Prediction")
                    pred_days = st.slider("Predict ahead (days):",1,90,30)
                    df_pred = hist_df.sort_index(ascending=True).tail(365).copy()
                    df_pred['Day_Num'] = np.arange(len(df_pred))
                    # Train model
                    X = df_pred['Day_Num'].values.reshape(-1,1)
                    y = df_pred['Close_PHP'].values
                    model = LinearRegression()
                    model.fit(X,y)
                    future_X = np.arange(len(df_pred), len(df_pred)+pred_days).reshape(-1,1)
                    future_y = model.predict(future_X)
                    future_dates = pd.date_range(start=df_pred.index[-1]+timedelta(days=1), periods=pred_days)
                    df_future = pd.DataFrame({'Predicted_Price_PHP':future_y}, index=future_dates)
                    st.line_chart(df_future['Predicted_Price_PHP'])

            # ---------- BUY / SELL ANALYSIS ----------
            # Optional: similar linear regression based analysis as before
            # Can be added similarly using df_pred and df_future

else:
    st.write("Enter a stock symbol above to get started.")
