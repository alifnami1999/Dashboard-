import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

# === Set Page Config ===
st.set_page_config(layout="wide")

# === Set Background Image from URL ===
def set_bg_from_url(image_url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .main {{
            background-color: rgba(ccddea);
            padding: 10px;
            border-radius: 10px;
        }}
        </style>
    """, unsafe_allow_html=True)

# ✅ Use Background Image URL
bg_url = "https://img.freepik.com/premium-photo/abstract-glowing-candlestick-forex-chart-with-index-grid-dark-background-invest-trade-finance-ans-stock-market-concept-3d-rendering_670147-7364.jpg"
set_bg_from_url(bg_url)

# === Main container ===
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    # === Title ===
    st.markdown("""
    <div style='border: 7px solid grey; padding: 20px; border-radius: 10px; background-color: ##ccddea;'>
        <h1 style='font-size: 60px; color: #1e3799; text-align: center;'>📈 FINTECHEDGE</h1>
        <p style='font-size: 30px; text-align: center;'>THE SHARPEST EDGE FOR EUR/USD 20-DAY AHEAD TREND PREDICTION</p>
    </div>
    """, unsafe_allow_html=True)

    # === Step 1: Upload CSV ===
    st.sidebar.header("Step 1: Upload CSV File")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip()
        df['Time'] = pd.to_datetime(df['Time'], dayfirst=True, errors='coerce')  # Handle DD/MM/YYYY too
        df = df.dropna(subset=['Time']).sort_values('Time').reset_index(drop=True)
        st.session_state.df_raw = df
        st.success("✅ CSV loaded successfully.")
        st.write("### Full Raw Dataset")
        st.dataframe(df, use_container_width=True)

    # === Step 2: Technical Indicators ===
    if 'df_raw' in st.session_state and st.sidebar.button("Step 2: Generate Technical Indicators"):
        df = st.session_state.df_raw.copy()

        def SMA(data, window=10): return data['Close'].rolling(window=window).mean()
        def WMA(data, window=10):
            weights = np.arange(1, window+1)
            return data['Close'].rolling(window).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
        def momentum(data, window=10): return data['Close'].diff(window)
        def stochastic_k(data, window=14):
            low_min = data['Low'].rolling(window).min()
            high_max = data['High'].rolling(window).max()
            return 100 * (data['Close'] - low_min) / (high_max - low_min)
        def stochastic_d(data, k_col='Stochastic_K%', window=3): return data[k_col].rolling(window).mean()
        def rsi(data, window=14):
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = -delta.where(delta < 0, 0).rolling(window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        def macd(data):
            ema12 = data['Close'].ewm(span=12).mean()
            ema26 = data['Close'].ewm(span=26).mean()
            line = ema12 - ema26
            signal = line.ewm(span=9).mean()
            return line, signal
        def williams_r(data, window=10):
            high = data['High'].rolling(window).max()
            low = data['Low'].rolling(window).min()
            return (high - data['Close']) / (high - low) * -100
        def ad_osc(data): return (data['High'].diff() - data['Low'].diff()).rolling(14).mean()
        def cci(data, window=14):
            tp = (data['High'] + data['Low'] + data['Close']) / 3
            sma = tp.rolling(window).mean()
            mad = (tp - sma).abs().rolling(window).mean()
            return (tp - sma) / (0.015 * mad)

        df['SMA'] = SMA(df)
        df['WMA'] = WMA(df)
        df['Momentum'] = momentum(df)
        df['Stochastic_K%'] = stochastic_k(df)
        df['Stochastic_D%'] = stochastic_d(df)
        df['RSI'] = rsi(df)
        df['MACD_Line'], df['MACD_Signal'] = macd(df)
        df['Williams_R'] = williams_r(df)
        df['A/D'] = ad_osc(df)
        df['CCI'] = cci(df)

        df = df.dropna().reset_index(drop=True)
        st.session_state.df_tech = df
        st.success("✅ Technical indicators calculated.")
        st.write("### Technical Indicator Dataset")
        st.dataframe(df, use_container_width=True)

    # === Step 3: Trend Conversion ===
    if 'df_tech' in st.session_state and st.sidebar.button("Step 3: Convert to Trend"):
        df = st.session_state.df_tech.copy()
        df['Trend'] = np.where(df['Close'].diff() > 0, 1, -1)
        df['SMA'] = np.where(df['Close'] > df['SMA'], 1, -1)
        df['WMA'] = np.where(df['Close'] > df['WMA'], 1, -1)
        df['Momentum'] = np.where(df['Momentum'] > 0, 1, -1)
        df['Stochastic_K%'] = np.where(df['Stochastic_K%'].diff() > 0, 1, -1)
        df['Stochastic_D%'] = np.where(df['Stochastic_D%'].diff() > 0, 1, -1)
        df['RSI'] = np.where(df['RSI'] < 30, 1,
                             np.where(df['RSI'] > 70, -1, np.where(df['RSI'].diff() > 0, 1, -1)))
        df['MACD_Line'] = np.where(df['MACD_Line'].diff() > 0, 1, -1)
        df['Williams_R'] = np.where(df['Williams_R'].diff() > 0, 1, -1)
        df['A/D'] = np.where(df['A/D'].diff() > 0, 1, -1)
        df['CCI'] = np.where(df['CCI'].diff() > 0, 1, -1)

        df = df.dropna().reset_index(drop=True)
        st.session_state.df_trend = df
        st.success("✅ Trend conversion done.")
        st.write("### Technical + Trend Dataset")
        st.dataframe(df, use_container_width=True)

    # === Step 4: Load Models ===
    try:
        xgb_loaded = joblib.load("xgb_best_model.pkl")
        rf_loaded = joblib.load("rf_best_model.pkl")
        svm_loaded = joblib.load("svm_best_model.pkl")
        scaler = joblib.load("scaler.pkl")

        st.session_state.models = {
            'XGBoost': xgb_loaded,
            'Random Forest': rf_loaded,
            'SVM': svm_loaded
        }
        st.session_state.scaler = scaler
        st.success("✅ Models and scaler loaded.")
    except Exception as e:
        st.error(f"❌ Could not load models: {e}")

    # === Step 5: Forecast ===
    if 'df_trend' in st.session_state and 'models' in st.session_state and 'scaler' in st.session_state:
        df = st.session_state.df_trend.copy()
        models = st.session_state.models
        scaler = st.session_state.scaler

        st.sidebar.header("Step 5: Forecast Settings")
        forecast_date = st.sidebar.date_input("Select Forecast Date")

        if st.sidebar.button("Run 20-Day Ahead Forecast"):
            df['Time'] = pd.to_datetime(df['Time'])
            forecast_date = pd.to_datetime(forecast_date)

            if forecast_date not in df['Time'].values:
                st.warning("⚠️ Forecast date not found. Using closest available.")
                forecast_date = df.iloc[(df['Time'] - forecast_date).abs().argsort().iloc[0]]['Time']

            forecast_idx = df[df['Time'] == forecast_date].index[0]
            input_cols = ['SMA', 'WMA', 'Momentum', 'Stochastic_K%', 'Stochastic_D%', 'RSI',
                          'MACD_Line', 'Williams_R', 'A/D', 'CCI']
            input_data = df.loc[[forecast_idx], input_cols]
            input_scaled = scaler.transform(input_data)

            forecast_offset = forecast_idx + 20
            pred_date = df.loc[forecast_offset, 'Time'].date() if forecast_offset < len(df) else (forecast_date + timedelta(days=20)).date()
            actual_trend = df.loc[forecast_offset, 'Trend'] if forecast_offset < len(df) else 'Out of Range'

            pred_results = {}
            for name, model in models.items():
                pred = model.predict(input_scaled)[0]
                pred_results[name] = "⬆️ +1" if pred == 1 else "⬇️ -1"

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<h2 style='color:#1e3799;'>📉 Forecast Results</h2>", unsafe_allow_html=True)

            st.markdown(f"""
            <div style='font-size:28px; padding:15px; background-color:#d3d3d3;
                        border:2px solid black; border-radius:10px; margin-bottom:10px; color:#1e3d59;'>
                📅 <b>Forecast Date:</b> {forecast_date.date()}<br>
                📆 <b>Predicted Date (+20 Days):</b> {pred_date}<br>
                📉 <b>Actual Trend:</b> {actual_trend}
            </div>
            """, unsafe_allow_html=True)

            for model_name, label in pred_results.items():
                st.markdown(f"""
                <div style='font-size:28px; padding:15px; background-color:#d3d3d3;
                            border:2px solid black; border-radius:10px; margin-bottom:10px; color:#1e3d59;'>
                    🤖 <b>{model_name} Prediction:</b> {label}
                </div>
                """, unsafe_allow_html=True)

            result_df = pd.DataFrame([{
                "Forecast Date": forecast_date.date(),
                "Predicted Date (+20)": pred_date,
                "XGBoost": pred_results['XGBoost'],
                "Random Forest": pred_results['Random Forest'],
                "SVM": pred_results['SVM'],
                "Actual Trend": actual_trend
            }])
            st.download_button("📥 Download Result", result_df.to_csv(index=False), file_name="forecast_result.csv")

    st.markdown("</div>", unsafe_allow_html=True)
