import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import timedelta, datetime
import requests

# MUST BE FIRST
st.set_page_config(page_title="Stock Price App", layout="wide")

# Initialize ALL session states at the very beginning
if 'show_latest' not in st.session_state:
    st.session_state.show_latest = False
if 'show_history' not in st.session_state:
    st.session_state.show_history = False
if 'show_chart' not in st.session_state:
    st.session_state.show_chart = False
if 'show_predict' not in st.session_state:
    st.session_state.show_predict = False
if 'show_buy' not in st.session_state:
    st.session_state.show_buy = False
if 'show_sell' not in st.session_state:
    st.session_state.show_sell = False

# Exchange rate USD to PHP
USD_TO_PHP = 59.14

st.title("Stock Price Analysis Application")
st.info(f"International Stocks: Yahoo Finance | Live Exchange Rate: 1 USD = ₱{USD_TO_PHP:.2f}")

# Helper text
st.caption("International stocks:** AAPL, TSLA, MSFT")

symbol = st.text_input("Enter stock symbol:")

# Buttons - Row 1
st.write("### Actions:")
col1, col2, col3 = st.columns(3)
if col1.button("Get Latest Price", use_container_width=True):
    st.session_state.show_latest = not st.session_state.show_latest
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col2.button("Show Historical Data", use_container_width=True):
    st.session_state.show_history = not st.session_state.show_history
    st.session_state.show_latest = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col3.button("Show Chart", use_container_width=True):
    st.session_state.show_chart = not st.session_state.show_chart
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False

# Button - Row 2
col4, col5, col6 = st.columns(3)
if col4.button("Predict Future Prices", use_container_width=True, type="primary"):
    st.session_state.show_predict = not st.session_state.show_predict
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_buy = False
    st.session_state.show_sell = False
    
if col5.button("Buy Recommendation", use_container_width=True):
    st.session_state.show_buy = not st.session_state.show_buy
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_sell = False
    
if col6.button("Sell Analysis", use_container_width=True):
    st.session_state.show_sell = not st.session_state.show_sell
    st.session_state.show_latest = False
    st.session_state.show_history = False
    st.session_state.show_chart = False
    st.session_state.show_predict = False
    st.session_state.show_buy = False

def is_pse_stock(symbol):
    """Check if this is a Philippine stock"""
    return symbol.upper().endswith('.PS')

@st.cache_data(ttl=3600)
def fetch_yahoo_data(symbol, period="1y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        
        if hist.empty:
            return None, None, False
        
        info = stock.info
        
        # Check if it's a Philippine stock
        is_pse = symbol.upper().endswith('.PS')
        
        return hist, info, is_pse
    except Exception as e:
        st.error(f"Error fetching Yahoo data: {e}")
        return None, None, False

def fetch_data_hybrid(symbol):
    """Smart fetch: automatically choose the right API"""
    # Add .PS if it looks like a PSE stock without suffix
    pse_stocks = ['JFC', 'SM', 'BDO', 'ALI', 'MBT', 'BPI', 'SMPH', 'MEG', 'TEL', 'GLO',
                  'AC', 'AGI', 'MER', 'DMC', 'URC', 'PGOLD', 'LTG', 'ICT', 'RLC', 'AEV',
                  'COL', 'BLOOM', 'CNPF', 'CEB', 'SMDC', 'FGEN', 'AP', 'EMI', 'HOUSE', 'CLI']
    
    if symbol.upper() in pse_stocks and not symbol.upper().endswith('.PS'):
        symbol = symbol.upper() + '.PS'
        st.info(f"🇵🇭 Auto-corrected to PSE format: {symbol}")
    
    if symbol.upper().endswith('.PS'):
        st.info(f"🇵🇭 Fetching Philippine stock: {symbol} - Using Yahoo Finance PSE data")
    else:
        st.info(f"🌍 Fetching international stock: {symbol} - Using Yahoo Finance")
    
    return fetch_yahoo_data(symbol)

if symbol:
    with st.spinner(f"Fetching data for {symbol.upper()}..."):
        hist_df, stock_info, is_pse = fetch_data_hybrid(symbol)

    if hist_df is not None and stock_info is not None:
        # Standardize column names
        hist_df = hist_df.reset_index()
        hist_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        hist_df.set_index('Date', inplace=True)
        hist_df = hist_df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Check if stock is in PHP (PSE stocks) or needs conversion
        if is_pse:
            # PSE stocks from Yahoo are already in PHP
            hist_df['Open_PHP'] = hist_df['Open']
            hist_df['High_PHP'] = hist_df['High']
            hist_df['Low_PHP'] = hist_df['Low']
            hist_df['Close_PHP'] = hist_df['Close']
        else:
            # Convert USD to PHP for international stocks
            hist_df['Open_PHP'] = hist_df['Open'] * USD_TO_PHP
            hist_df['High_PHP'] = hist_df['High'] * USD_TO_PHP
            hist_df['Low_PHP'] = hist_df['Low'] * USD_TO_PHP
            hist_df['Close_PHP'] = hist_df['Close'] * USD_TO_PHP

        hist_df = hist_df.sort_index(ascending=False)

        # No warning needed - Yahoo Finance has full historical data for PSE

        # LATEST PRICE
        if st.session_state.show_latest:
            st.subheader("Latest Stock Price")

            latest_date = hist_df.index[0]
            latest_row = hist_df.loc[latest_date]

            if len(hist_df) > 1:
                prev_close = hist_df.iloc[1]["Close_PHP"]
                change = latest_row["Close_PHP"] - prev_close
                change_pct = (change / prev_close) * 100
                change_color = "green" if change >= 0 else "red"
                change_symbol = "▲" if change >= 0 else "▼"
            else:
                change = 0
                change_pct = 0
                change_color = "gray"
                change_symbol = "●"

            company_name = stock_info.get('longName', symbol.upper())

            st.markdown(
                f"""
                <div style="padding: 20px; background-color: #f7f7f7; border-radius: 10px; border: 1px solid #ddd;">
                    <h3 style="margin-top:0;">Latest data for <b>{company_name}</b> ({symbol.upper()})</h3>
                    <p style="font-size: 14px; color: #666;">Date: {latest_date.strftime('%Y-%m-%d %H:%M')}</p>
                    <p style="font-size: 24px; font-weight: bold; color: {change_color};">
                        ₱{latest_row["Close_PHP"]:.2f} 
                        <span style="font-size: 18px;">{change_symbol} {change:.2f} ({change_pct:+.2f}%)</span>
                    </p>
                    <p style="font-size: 12px; color: #999;">{'Already in PHP' if is_pse else f'Exchange Rate: 1 USD = ₱{USD_TO_PHP:.2f}'}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            if is_pse:
                latest_df = pd.DataFrame({
                    "Metric": ["Open", "High", "Low", "Close", "Volume"],
                    "Value (PHP)": [
                        f"₱{latest_row['Open_PHP']:.2f}",
                        f"₱{latest_row['High_PHP']:.2f}",
                        f"₱{latest_row['Low_PHP']:.2f}",
                        f"₱{latest_row['Close_PHP']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ]
                })
            else:
                latest_df = pd.DataFrame({
                    "Metric": ["Open", "High", "Low", "Close", "Volume"],
                    "Value (PHP)": [
                        f"₱{latest_row['Open_PHP']:.2f}",
                        f"₱{latest_row['High_PHP']:.2f}",
                        f"₱{latest_row['Low_PHP']:.2f}",
                        f"₱{latest_row['Close_PHP']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ],
                    "Value (USD)": [
                        f"${latest_row['Open']:.2f}",
                        f"${latest_row['High']:.2f}",
                        f"${latest_row['Low']:.2f}",
                        f"${latest_row['Close']:.2f}",
                        f"{int(latest_row['Volume']):,}"
                    ]
                })

            st.table(latest_df)
            
            # VOLATILITY ANALYSIS
            st.write("---")
            st.write("### 📊 Volatility Analysis")
            
            # Calculate various volatility metrics
            # 1. Standard Deviation (30-day)
            recent_30d = hist_df.head(min(30, len(hist_df)))
            daily_returns = recent_30d['Close_PHP'].pct_change().dropna()
            std_dev_30d = daily_returns.std() * 100  # Convert to percentage
            
            # 2. Historical Volatility (Annualized)
            historical_volatility = daily_returns.std() * np.sqrt(252) * 100  # 252 trading days/year
            
            # 3. Average True Range (ATR) - last 14 days
            recent_14d = hist_df.head(min(14, len(hist_df)))
            high_low = recent_14d['High_PHP'] - recent_14d['Low_PHP']
            atr = high_low.mean()
            atr_percent = (atr / recent_14d['Close_PHP'].mean()) * 100
            
            # 4. Volatility Classification
            if historical_volatility < 15:
                volatility_class = "Low Volatility"
                volatility_color = "green"
                volatility_desc = "Stable stock with small price movements"
            elif historical_volatility < 30:
                volatility_class = "Moderate Volatility"
                volatility_color = "orange"
                volatility_desc = "Average volatility, typical for most stocks"
            elif historical_volatility < 50:
                volatility_class = "High Volatility"
                volatility_color = "red"
                volatility_desc = "Significant price swings, higher risk"
            else:
                volatility_class = "Very High Volatility"
                volatility_color = "darkred"
                volatility_desc = "Extreme price movements, very risky"
            
            # Display volatility metrics
            col_vol1, col_vol2, col_vol3, col_vol4 = st.columns(4)
            
            with col_vol1:
                st.metric("Daily Volatility (30d)", f"{std_dev_30d:.2f}%")
            with col_vol2:
                st.metric("Annual Volatility", f"{historical_volatility:.2f}%")
            with col_vol3:
                st.metric("Avg True Range (14d)", f"₱{atr:.2f}")
            with col_vol4:
                st.metric("ATR %", f"{atr_percent:.2f}%")
            
            # Volatility classification
            st.markdown(
                f"""
                <div style="padding: 15px; background-color: #f0f0f0; border-radius: 8px; border-left: 5px solid {volatility_color};">
                    <h4 style="margin-top:0; color: {volatility_color};">Classification: {volatility_class}</h4>
                    <p style="margin-bottom:0;">{volatility_desc}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Volatility interpretation
            with st.expander("📖 Understanding Volatility Metrics"):
                st.write("""
                **Daily Volatility (30-day):** 
                - Measures day-to-day price fluctuations over the last 30 days
                - Higher = More unpredictable daily price changes
                
                **Annual Volatility (Annualized):**
                - Standard measure used by investors
                - < 15%: Low volatility (stable stocks like utilities)
                - 15-30%: Moderate (typical large-cap stocks)
                - 30-50%: High (growth stocks, tech)
                - > 50%: Very high (speculative, penny stocks)
                
                **Average True Range (ATR):**
                - Shows the average price range per day
                - Higher ATR = Bigger daily swings
                - Useful for setting stop-loss orders
                
                **ATR %:**
                - ATR as a percentage of stock price
                - Easier to compare across different stocks
                - 2-5%: Low volatility
                - 5-10%: Moderate volatility
                - 10%+: High volatility
                """)
            
            with st.expander("Company Information"):
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                    st.write(f"**Industry:** {stock_info.get('industry', 'N/A')}")
                    st.write(f"**Market Cap:** {stock_info.get('marketCap', 'N/A')}")
                with col_info2:
                    st.write(f"**52 Week High:** {stock_info.get('fiftyTwoWeekHigh', 'N/A')}")
                    st.write(f"**52 Week Low:** {stock_info.get('fiftyTwoWeekLow', 'N/A')}")
                    st.write(f"**Website:** {stock_info.get('website', 'N/A')}")

        # HISTORICAL DATA
        if st.session_state.show_history:
            st.subheader("Historical Stock Prices")
            
            if len(hist_df) >= 30:
                days = st.selectbox("Select time period:", [30, 60, 90, 180, 365, "All"], index=1)
                
                if days == "All":
                    display_df = hist_df
                else:
                    display_df = hist_df.head(days)
            else:
                display_df = hist_df
                st.info(f"Showing all available data ({len(hist_df)} days)")
            
            display_columns = ['Open_PHP', 'High_PHP', 'Low_PHP', 'Close_PHP', 'Volume']
            display_df_show = display_df[display_columns].copy()
            display_df_show.columns = ['Open (PHP)', 'High (PHP)', 'Low (PHP)', 'Close (PHP)', 'Volume']
            
            st.dataframe(display_df_show.style.format({
                "Open (PHP)": "₱{:.2f}",
                "High (PHP)": "₱{:.2f}",
                "Low (PHP)": "₱{:.2f}",
                "Close (PHP)": "₱{:.2f}",
                "Volume": "{:,.0f}"
            }), use_container_width=True)
            
            csv = display_df_show.to_csv()
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"{symbol}_stock_data.csv",
                mime="text/csv"
            )

        # CHARTS
        if st.session_state.show_chart:
            st.subheader("Stock Charts")

            if len(hist_df) >= 30:
                days = st.selectbox("Chart time period:", [30, 60, 90, 180, 365, "All"], index=1, key="chart_days")
                
                if days == "All":
                    chart_df = hist_df.sort_index(ascending=True)
                else:
                    chart_df = hist_df.head(days).sort_index(ascending=True)
            else:
                chart_df = hist_df.sort_index(ascending=True)

            st.write("### Closing Price Trend")
            st.line_chart(chart_df["Close_PHP"])

            st.write("### Candlestick Chart")
            candle_fig = go.Figure(data=[go.Candlestick(
                x=chart_df.index,
                open=chart_df["Open_PHP"],
                high=chart_df["High_PHP"],
                low=chart_df["Low_PHP"],
                close=chart_df["Close_PHP"],
                name=symbol.upper()
            )])
            candle_fig.update_layout(
                height=600,
                xaxis_title="Date",
                yaxis_title="Price (PHP)",
                hovermode="x unified"
            )
            st.plotly_chart(candle_fig, use_container_width=True)

            st.write("### Volume Chart")
            st.bar_chart(chart_df["Volume"])

        # PREDICT
        if st.session_state.show_predict:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for prediction. Need at least 30 days of data.")
            else:
                st.subheader("Stock Price Prediction (Linear Regression)")
                
                st.warning("**Disclaimer:** This is a simple linear regression model for educational purposes only.")
                
                col_pred1, col_pred2 = st.columns(2)
                with col_pred1:
                    max_training = min(len(hist_df), 365)
                    training_days = st.slider("Training period (days):", 30, max_training, min(90, max_training))
                with col_pred2:
                    predict_days = st.slider("Predict ahead (days):", 1, 90, 30)
                
                pred_df = hist_df.sort_index(ascending=True).tail(training_days).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                # Train PRICE model
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y_price = pred_df['Close_PHP'].values
                
                model_price = LinearRegression()
                model_price.fit(X, y_price)
                pred_df['Predicted_Price'] = model_price.predict(X)
                
                # Train VOLUME model
                y_volume = pred_df['Volume'].values
                model_volume = LinearRegression()
                model_volume.fit(X, y_volume)
                pred_df['Predicted_Volume'] = model_volume.predict(X)
                
                # Predict future prices
                future_days = np.arange(len(pred_df), len(pred_df) + predict_days).reshape(-1, 1)
                future_prices = model_price.predict(future_days)
                future_volumes = model_volume.predict(future_days)
                
                # Ensure volumes are non-negative
                future_volumes = np.maximum(future_volumes, 0)
                
                last_date = pred_df.index[-1]
                future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=predict_days, freq='D')
                
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted_Price_PHP': future_prices,
                    'Predicted_Volume': future_volumes
                })
                future_df.set_index('Date', inplace=True)
                
                from sklearn.metrics import r2_score, mean_absolute_error
                r2_price = r2_score(y_price, pred_df['Predicted_Price'])
                mae_price = mean_absolute_error(y_price, pred_df['Predicted_Price'])
                
                r2_volume = r2_score(y_volume, pred_df['Predicted_Volume'])
                mae_volume = mean_absolute_error(y_volume, pred_df['Predicted_Volume'])
                
                # Display Price metrics
                st.write("### 📈 Price Prediction Metrics")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("R² Score", f"{r2_price:.4f}")
                with col_m2:
                    st.metric("Mean Abs Error", f"₱{mae_price:.2f}")
                with col_m3:
                    st.metric("Current Price", f"₱{pred_df['Close_PHP'].iloc[-1]:.2f}")
                with col_m4:
                    predicted_price = future_prices[-1]
                    price_change = predicted_price - pred_df['Close_PHP'].iloc[-1]
                    st.metric(f"Predicted ({predict_days}d)", f"₱{predicted_price:.2f}", f"{price_change:+.2f}")
                
                if r2_price > 0.7:
                    st.success(f"Strong linear trend detected (R² = {r2_price:.4f})")
                elif r2_price > 0.4:
                    st.info(f"Moderate linear trend (R² = {r2_price:.4f})")
                else:
                    st.warning(f"Weak linear trend (R² = {r2_price:.4f})")
                
                # Display Volume metrics
                st.write("### 📊 Volume Prediction Metrics")
                col_v1, col_v2, col_v3, col_v4 = st.columns(4)
                with col_v1:
                    st.metric("R² Score", f"{r2_volume:.4f}")
                with col_v2:
                    st.metric("Mean Abs Error", f"{mae_volume:,.0f}")
                with col_v3:
                    st.metric("Current Volume", f"{int(pred_df['Volume'].iloc[-1]):,}")
                with col_v4:
                    predicted_volume = future_volumes[-1]
                    volume_change = predicted_volume - pred_df['Volume'].iloc[-1]
                    st.metric(f"Predicted ({predict_days}d)", f"{int(predicted_volume):,}", f"{volume_change:+,.0f}")
                
                if r2_volume > 0.5:
                    st.success(f"Moderate volume trend detected (R² = {r2_volume:.4f})")
                elif r2_volume > 0.3:
                    st.info(f"Weak volume trend (R² = {r2_volume:.4f})")
                else:
                    st.warning(f"Very weak volume trend (R² = {r2_volume:.4f}) - Volume predictions may be unreliable")
                
                st.write("---")
                
                st.write("### Future Price & Volume Predictions Table")
                
                # Format the future predictions table
                future_display = future_df.copy()
                future_display['Predicted_Price_PHP'] = future_display['Predicted_Price_PHP'].apply(lambda x: f"₱{x:.2f}")
                future_display['Predicted_Volume'] = future_display['Predicted_Volume'].apply(lambda x: f"{int(x):,}")
                if not is_pse:
                    future_display['Predicted_Price_USD'] = (future_df['Predicted_Price_PHP'] / USD_TO_PHP).apply(lambda x: f"${x:.2f}")
                future_display.index = future_display.index.strftime('%Y-%m-%d')
                future_display.index.name = 'Date'
                
                st.dataframe(future_display, use_container_width=True, height=400)
                
                csv_pred = future_df.to_csv()
                st.download_button(
                    label="Download Predictions CSV",
                    data=csv_pred,
                    file_name=f"{symbol}_predictions.csv",
                    mime="text/csv"
                )
                
                st.write("---")
                st.write("### Price Prediction Chart")
                
                fig_pred = go.Figure()
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Close_PHP'],
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Predicted_Price'],
                    mode='lines',
                    name='Fitted Line',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df['Predicted_Price_PHP'],
                    mode='lines+markers',
                    name='Future Prediction',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=6)
                ))
                
                fig_pred.update_layout(
                    height=600,
                    xaxis_title="Date",
                    yaxis_title="Price (PHP)",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Volume Prediction Chart
                st.write("### Volume Prediction Chart")
                
                fig_vol = go.Figure()
                
                fig_vol.add_trace(go.Bar(
                    x=pred_df.index,
                    y=pred_df['Volume'],
                    name='Actual Volume',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                fig_vol.add_trace(go.Scatter(
                    x=pred_df.index,
                    y=pred_df['Predicted_Volume'],
                    mode='lines',
                    name='Fitted Volume',
                    line=dict(color='orange', width=2, dash='dash')
                ))
                
                fig_vol.add_trace(go.Scatter(
                    x=future_df.index,
                    y=future_df['Predicted_Volume'],
                    mode='lines+markers',
                    name='Future Volume Prediction',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(size=6)
                ))
                
                fig_vol.update_layout(
                    height=600,
                    xaxis_title="Date",
                    yaxis_title="Volume",
                    hovermode="x unified",
                    showlegend=True
                )
                
                st.plotly_chart(fig_vol, use_container_width=True)
                
                st.write("### Model Details")
                
                slope_price = model_price.coef_[0]
                intercept_price = model_price.intercept_
                st.code(f"Price = ₱{intercept_price:.2f} + ₱{slope_price:.2f} × Day")
                
                slope_volume = model_volume.coef_[0]
                intercept_volume = model_volume.intercept_
                st.code(f"Volume = {intercept_volume:,.0f} + {slope_volume:,.0f} × Day")
                
                trend_direction_price = "upward" if slope_price > 0 else "downward"
                trend_direction_volume = "increasing" if slope_volume > 0 else "decreasing"
                
                st.write(f"**Price Trend:** {trend_direction_price} trend of **₱{abs(slope_price):.2f} per day**.")
                st.write(f"**Volume Trend:** {trend_direction_volume} trend of **{abs(slope_volume):,.0f} shares per day**.")

        # BUY RECOMMENDATION
        if st.session_state.show_buy:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for recommendation. Need at least 30 days.")
            else:
                st.subheader("Buy Recommendation Analysis")
                
                st.info("This analysis uses linear regression prediction for both price and volume.")
                
                col_buy1, col_buy2 = st.columns(2)
                with col_buy1:
                    num_shares = st.number_input("How many shares do you plan to buy?", min_value=1, value=100, step=1)
                with col_buy2:
                    investment_horizon = st.slider("Investment time horizon (days):", 7, 90, 30)
                
                current_price_php = hist_df.iloc[0]['Close_PHP']
                current_volume = hist_df.iloc[0]['Volume']
                
                pred_df = hist_df.sort_index(ascending=True).tail(min(90, len(hist_df))).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                # Train price model
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y_price = pred_df['Close_PHP'].values
                model_price = LinearRegression()
                model_price.fit(X, y_price)
                
                # Train volume model
                y_volume = pred_df['Volume'].values
                model_volume = LinearRegression()
                model_volume.fit(X, y_volume)
                
                future_day = np.array([[len(pred_df) + investment_horizon]])
                predicted_price_php = model_price.predict(future_day)[0]
                predicted_volume = max(0, model_volume.predict(future_day)[0])
                
                price_change = predicted_price_php - current_price_php
                price_change_pct = (price_change / current_price_php) * 100
                
                volume_change = predicted_volume - current_volume
                volume_change_pct = (volume_change / current_volume) * 100 if current_volume > 0 else 0
                
                total_investment = num_shares * current_price_php
                potential_value = num_shares * predicted_price_php
                potential_profit = potential_value - total_investment
                
                pred_df['Predicted_Price'] = model_price.predict(X)
                pred_df['Predicted_Volume'] = model_volume.predict(X)
                
                from sklearn.metrics import r2_score
                r2_price = r2_score(y_price, pred_df['Predicted_Price'])
                r2_volume = r2_score(y_volume, pred_df['Predicted_Volume'])
                
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    st.metric("Current Price", f"₱{current_price_php:.2f}")
                with col_p2:
                    st.metric(f"Predicted Price ({investment_horizon}d)", f"₱{predicted_price_php:.2f}", f"{price_change_pct:+.2f}%")
                with col_p3:
                    st.metric("Price R² Score", f"{r2_price:.4f}")
                
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.metric("Current Volume", f"{int(current_volume):,}")
                with col_v2:
                    st.metric(f"Predicted Volume ({investment_horizon}d)", f"{int(predicted_volume):,}", f"{volume_change_pct:+.2f}%")
                with col_v3:
                    st.metric("Volume R² Score", f"{r2_volume:.4f}")
                
                st.write("---")
                st.write("### Investment Analysis")
                
                investment_df = pd.DataFrame({
                    "Item": [
                        "Number of Shares",
                        "Current Price per Share",
                        "Current Volume",
                        "Total Investment",
                        f"Predicted Price ({investment_horizon}d)",
                        f"Predicted Volume ({investment_horizon}d)",
                        "Predicted Value",
                        "Potential Profit/Loss",
                        "Return (%)"
                    ],
                    "Amount": [
                        f"{num_shares:,}",
                        f"₱{current_price_php:.2f}",
                        f"{int(current_volume):,}",
                        f"₱{total_investment:,.2f}",
                        f"₱{predicted_price_php:.2f}",
                        f"{int(predicted_volume):,}",
                        f"₱{potential_value:,.2f}",
                        f"₱{potential_profit:+,.2f}",
                        f"{price_change_pct:+.2f}%"
                    ]
                })
                
                st.table(investment_df)
                
                st.write("### Recommendation")
                
                is_uptrend = price_change > 0
                is_reliable = r2_price > 0.5
                is_significant = price_change_pct > 5
                volume_increasing = volume_change > 0
                
                if is_uptrend and is_reliable and is_significant:
                    st.success("✅ STRONG BUY")
                    st.write(f"- Predicted {price_change_pct:.2f}% price increase")
                    st.write(f"- Good price reliability (R² = {r2_price:.4f})")
                    if volume_increasing:
                        st.write(f"- Volume expected to increase by {volume_change_pct:.2f}% (bullish signal)")
                    else:
                        st.write(f"- Volume expected to decrease by {abs(volume_change_pct):.2f}% (caution)")
                elif is_uptrend and is_reliable:
                    st.info("ℹ️ MODERATE BUY")
                    st.write(f"- Predicted {price_change_pct:.2f}% price increase")
                    st.write(f"- Volume trend: {volume_change_pct:+.2f}%")
                elif is_uptrend:
                    st.warning("⚠️ CAUTIOUS - LOW CONFIDENCE")
                    st.write(f"- Low price reliability (R² = {r2_price:.4f})")
                else:
                    st.error("❌ NOT RECOMMENDED")
                    st.write(f"- Predicted {price_change_pct:.2f}% price decrease")
                    
                # Volume liquidity warning
                if predicted_volume < current_volume * 0.5:
                    st.warning("⚠️ **Liquidity Warning:** Predicted volume drop of more than 50% may indicate reduced trading activity.")

        # SELL ANALYSIS
        if st.session_state.show_sell:
            if len(hist_df) < 30:
                st.error("❌ Insufficient historical data for analysis. Need at least 30 days.")
            else:
                st.subheader("Sell Analysis & Profit Calculator")
                
                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    shares_owned = st.number_input("Shares owned:", min_value=1, value=100, step=1)
                with col_s2:
                    purchase_price = st.number_input("Purchase price (PHP):", min_value=0.01, value=1000.0, step=0.01)
                with col_s3:
                    hold_days = st.slider("Hold before selling (days):", 7, 90, 30)
                
                current_price_php = hist_df.iloc[0]['Close_PHP']
                current_volume = hist_df.iloc[0]['Volume']
                
                total_purchase = shares_owned * purchase_price
                total_current = shares_owned * current_price_php
                current_profit = total_current - total_purchase
                current_profit_pct = (current_profit / total_purchase) * 100
                
                pred_df = hist_df.sort_index(ascending=True).tail(min(90, len(hist_df))).copy()
                pred_df['Day_Num'] = np.arange(len(pred_df))
                
                # Train price model
                X = pred_df['Day_Num'].values.reshape(-1, 1)
                y_price = pred_df['Close_PHP'].values
                model_price = LinearRegression()
                model_price.fit(X, y_price)
                
                # Train volume model
                y_volume = pred_df['Volume'].values
                model_volume = LinearRegression()
                model_volume.fit(X, y_volume)
                
                future_day = np.array([[len(pred_df) + hold_days]])
                predicted_price = model_price.predict(future_day)[0]
                predicted_volume = max(0, model_volume.predict(future_day)[0])
                
                total_predicted = shares_owned * predicted_price
                predicted_profit = total_predicted - total_purchase
                predicted_profit_pct = (predicted_profit / total_purchase) * 100
                
                additional = predicted_profit - current_profit
                
                volume_change = predicted_volume - current_volume
                volume_change_pct = (volume_change / current_volume) * 100 if current_volume > 0 else 0
                
                st.write("### Current Position")
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Purchase", f"₱{purchase_price:.2f}")
                with col_m2:
                    st.metric("Current", f"₱{current_price_php:.2f}")
                with col_m3:
                    st.metric("Profit/Loss", f"₱{current_profit:,.2f}", f"{current_profit_pct:+.2f}%")
                with col_m4:
                    st.metric(f"Predicted ({hold_days}d)", f"₱{predicted_price:.2f}")
                
                st.write("### Volume Analysis")
                col_v1, col_v2 = st.columns(2)
                with col_v1:
                    st.metric("Current Volume", f"{int(current_volume):,}")
                with col_v2:
                    st.metric(f"Predicted Volume ({hold_days}d)", f"{int(predicted_volume):,}", f"{volume_change_pct:+.2f}%")
                
                st.write("---")
                st.write("### Analysis")
                
                analysis_df = pd.DataFrame({
                    "Scenario": ["Sell Now", f"Sell in {hold_days}d"],
                    "Price": [f"₱{current_price_php:.2f}", f"₱{predicted_price:.2f}"],
                    "Volume": [f"{int(current_volume):,}", f"{int(predicted_volume):,}"],
                    "Total Value": [f"₱{total_current:,.2f}", f"₱{total_predicted:,.2f}"],
                    "Profit": [f"₱{current_profit:+,.2f}", f"₱{predicted_profit:+,.2f}"],
                    "Return": [f"{current_profit_pct:+.2f}%", f"{predicted_profit_pct:+.2f}%"]
                })
                
                st.table(analysis_df)
                
                if additional > 0:
                    st.info(f"💡 Potential additional gain: ₱{additional:,.2f}")
                else:
                    st.warning(f"⚠️ Potential additional loss: ₱{abs(additional):,.2f}")
                
                st.write("### Recommendation")
                
                has_profit = current_profit > 0
                will_increase = predicted_price > current_price_php
                volume_increasing = volume_change > 0
                
                if has_profit and current_profit_pct > 10 and not will_increase:
                    st.success("✅ STRONG SELL - Lock in profits")
                    if not volume_increasing:
                        st.write("- Volume declining suggests weakening interest")
                elif has_profit and will_increase:
                    st.info("ℹ️ HOLD - More gains expected")
                    if volume_increasing:
                        st.write("- Rising volume supports price increase")
                    else:
                        st.write("- Declining volume may limit upside")
                elif not has_profit and will_increase:
                    st.warning("⚠️ HOLD - Wait for recovery")
                else:
                    st.error("❌ MINIMIZE LOSS")
                    
                # Liquidity warning for selling
                if predicted_volume < current_volume * 0.5:
                    st.warning("⚠️ **Liquidity Warning:** Significant volume drop predicted. May be harder to sell at desired price.")

    else:
        st.error(f"Unable to fetch data for: {symbol.upper()}")
        st.info("**Tips:**\n- For international stocks: Use symbols like AAPL, TSLA, GOOGL\n- For Philippine stocks: Add .PS suffix like JFC.PS, SM.PS, BDO.PS, or just type JFC, SM, BDO (auto-adds .PS)")
else:
    st.write("Enter a stock symbol above to get started.")


