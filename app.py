"""
Tesla Stock Price Prediction & Forecasting App
------------------------------------------------
Run with:  streamlit run app.py

Required files in the same directory:
    - TESLA_RNN_BEST_MODEL.keras   (trained model)

Note: No pre-saved scaler is used. The scaler is fit on-the-fly using
the uploaded CSV data (StandardScaler).
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="Tesla Stock Price Predictor", layout="wide")
st.title("📈 Tesla (TSLA) Stock Price Prediction & Forecast")
st.markdown(
    "This app uses a trained **deep learning model** to predict Tesla's "
    "closing price and generate a multi-day forecast based on historical "
    "OHLC data."
)

SEQ_LEN = 60
FEATURES = ["Open", "High", "Low", "Close"]
CLOSE_IDX = FEATURES.index("Close")

# NOTE: The trained model expects only the Close price as input (1 feature
# per timestep), not all 4 OHLC columns. Adjust MODEL_FEATURES if your model
# was trained on a different set of columns.
MODEL_FEATURES = ["Close"]
N_MODEL_FEATURES = len(MODEL_FEATURES)


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = load_model("TESLA_RNN_BEST_MODEL.keras", compile=False)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def inverse_transform_close(scaled_close, scaler, n_features=1, close_idx=0):
    """Inverse-transform a 1D array of scaled Close values back to dollars."""
    scaled_close = np.array(scaled_close).reshape(-1, 1)
    dummy = np.zeros((len(scaled_close), n_features))
    dummy[:, close_idx] = scaled_close.flatten()
    return scaler.inverse_transform(dummy)[:, close_idx]


def forecast_future(model, scaler, last_sequence, n_days, close_idx=0, n_features=1):
    """
    Recursively forecast `n_days` ahead.
    last_sequence: scaled array of shape (SEQ_LEN, n_features) - most recent window
    Returns: list of predicted Close prices (in original $ scale)
    """
    seq = last_sequence.copy()
    preds_scaled = []

    for _ in range(n_days):
        x_input = seq.reshape(1, seq.shape[0], seq.shape[1])
        pred_scaled = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(pred_scaled)

        # Build the next row: assume Open/High/Low ~ predicted Close
        # (a simplifying assumption since we only predict Close)
        next_row = seq[-1].copy()
        for i in range(n_features):
            next_row[i] = pred_scaled

        seq = np.vstack([seq[1:], next_row])

    return inverse_transform_close(preds_scaled, scaler, n_features, close_idx)


# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
try:
    model = load_artifacts()
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()


# ------------------------------------------------------------------
# Data upload
# ------------------------------------------------------------------
st.header("1. Upload Historical Data")
st.markdown(
    "Upload a CSV file containing at least the last **60 rows** with columns: "
    "`Open, High, Low, Close` (most recent date last)."
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if not all(col in df.columns for col in FEATURES):
        st.error(f"CSV must contain columns: {FEATURES}")
        st.stop()

    if len(df) < SEQ_LEN:
        st.error(f"Need at least {SEQ_LEN} rows of data. Uploaded file has {len(df)}.")
        st.stop()

    # Try to parse a Date column for nicer x-axes (optional)
    date_col = None
    for candidate in ["Date", "date", "Datetime", "datetime"]:
        if candidate in df.columns:
            try:
                df[candidate] = pd.to_datetime(df[candidate])
                date_col = candidate
                break
            except Exception:
                pass

    x_axis = df[date_col] if date_col else df.index

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.tail(10), use_container_width=True)

    # ==================================================================
    #  EXPLORATORY DATA ANALYSIS (EDA)
    # ==================================================================
    st.header("📊 Exploratory Data Analysis")

    # ---- Summary metric cards ----
    st.subheader("Dataset Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Avg Close", f"${df['Close'].mean():.2f}")
    c3.metric("Min Close", f"${df['Close'].min():.2f}")
    c4.metric("Max Close", f"${df['Close'].max():.2f}")
    c5.metric("Volatility (Std)", f"${df['Close'].std():.2f}")

    # ---- Descriptive statistics ----
    with st.expander("🔍 View Descriptive Statistics"):
        st.dataframe(
            df[FEATURES].describe().T.style.background_gradient(cmap="Blues"),
            use_container_width=True,
        )

    # ---- Candlestick chart ----
    st.subheader("Candlestick Chart (OHLC)")
    fig_candle = go.Figure(
        data=[
            go.Candlestick(
                x=x_axis,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                increasing_line_color="#00CC96",
                decreasing_line_color="#EF553B",
            )
        ]
    )
    fig_candle.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=450,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_candle, use_container_width=True)

    # ---- Close price trend with moving averages ----
    st.subheader("Close Price Trend with Moving Averages")
    ma_df = df.copy()
    ma_df["MA20"] = ma_df["Close"].rolling(window=20).mean()
    ma_df["MA50"] = ma_df["Close"].rolling(window=50).mean()

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=x_axis, y=ma_df["Close"], name="Close",
        line=dict(color="#636EFA", width=2)
    ))
    fig_trend.add_trace(go.Scatter(
        x=x_axis, y=ma_df["MA20"], name="MA 20",
        line=dict(color="#FFA15A", width=1.5, dash="dot")
    ))
    fig_trend.add_trace(go.Scatter(
        x=x_axis, y=ma_df["MA50"], name="MA 50",
        line=dict(color="#19D3F3", width=1.5, dash="dot")
    ))
    fig_trend.update_layout(
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # ---- Two-column row: distribution + correlation heatmap ----
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Close Price Distribution")
        fig_hist = px.histogram(
            df, x="Close", nbins=40,
            color_discrete_sequence=["#636EFA"],
            template="plotly_dark",
        )
        fig_hist.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_b:
        st.subheader("Feature Correlation Heatmap")
        corr = df[FEATURES + (["Volume"] if "Volume" in df.columns else [])].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f",
            linewidths=0.5, ax=ax_corr, cbar_kws={"shrink": 0.8}
        )
        fig_corr.patch.set_alpha(0)
        st.pyplot(fig_corr)

    # ---- Daily returns ----
    st.subheader("Daily Returns (%)")
    returns_df = df.copy()
    returns_df["Daily Return"] = returns_df["Close"].pct_change() * 100

    fig_ret = px.area(
        returns_df, x=x_axis, y="Daily Return",
        template="plotly_dark",
        color_discrete_sequence=["#00CC96"],
    )
    fig_ret.add_hline(y=0, line_dash="dash", line_color="grey")
    fig_ret.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_ret, use_container_width=True)

    # ---- Volume chart (if available) ----
    if "Volume" in df.columns:
        st.subheader("Trading Volume")
        fig_vol = px.bar(
            df, x=x_axis, y="Volume",
            template="plotly_dark",
            color_discrete_sequence=["#AB63FA"],
        )
        fig_vol.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_vol, use_container_width=True)

    st.divider()

    # ==================================================================
    #  MODEL PREP
    # ==================================================================

    # Fit scaler on the Close column only (matches model's expected input)
    scaler = StandardScaler()
    data = df[MODEL_FEATURES].values  # shape (n, 1) -> Close only
    data_scaled = scaler.fit_transform(data)

    # ------------------------------------------------------------------
    # Single next-day prediction
    # ------------------------------------------------------------------
    st.header("🔮 2. Next-Day Prediction")

    last_seq = data_scaled[-SEQ_LEN:]
    x_input = last_seq.reshape(1, SEQ_LEN, N_MODEL_FEATURES)

    pred_scaled = model.predict(x_input, verbose=0)
    next_day_pred = inverse_transform_close(pred_scaled, scaler)[0]

    last_close = df["Close"].iloc[-1]
    change = next_day_pred - last_close
    pct_change = (change / last_close) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Last Close Price", f"${last_close:.2f}")
    col2.metric("Predicted Next Close", f"${next_day_pred:.2f}", f"{change:+.2f}")
    col3.metric("% Change", f"{pct_change:+.2f}%")

    # ------------------------------------------------------------------
    # Multi-day forecast
    # ------------------------------------------------------------------
    st.header("📅 3. Multi-Day Forecast")

    n_days = st.slider("Select number of days to forecast", 1, 30, 10)

    if st.button("🚀 Generate Forecast"):
        with st.spinner("Generating forecast..."):
            forecast_prices = forecast_future(
                model, scaler, last_seq, n_days,
                close_idx=0, n_features=N_MODEL_FEATURES
            )

        forecast_df = pd.DataFrame({
            "Day": [f"Day +{i+1}" for i in range(n_days)],
            "Predicted Close ($)": forecast_prices
        })

        st.subheader(f"Forecast for the Next {n_days} Days")
        st.dataframe(forecast_df, use_container_width=True)

        # Plot historical + forecast
        history_len = 60
        hist_prices = df["Close"].values[-history_len:]

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=list(range(len(hist_prices))), y=hist_prices,
            name="Historical Close", line=dict(color="#636EFA", width=2)
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(range(len(hist_prices) - 1, len(hist_prices) + n_days)),
            y=[hist_prices[-1]] + list(forecast_prices),
            name="Forecast", line=dict(color="#FFA15A", width=2, dash="dash"),
            mode="lines+markers"
        ))
        fig_fc.update_layout(
            template="plotly_dark",
            title="Historical Prices and Forecast",
            xaxis_title="Time Step",
            yaxis_title="Price ($)",
            height=450,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.info(
            "⚠️ **Disclaimer:** Multi-day forecasts are generated recursively — "
            "each prediction is fed back as input for the next, so errors compound "
            "over time. Forecasts assume Open/High/Low track the predicted Close "
            "value, which is a simplification. Use this for trend indication only, "
            "not as financial advice."
        )

else:
    st.info("👆 Please upload a CSV file with historical OHLC data to get started.")

    st.markdown(
        """
        ### Expected CSV Format
        | Date       | Open   | High   | Low    | Close  | Volume    |
        |------------|--------|--------|--------|--------|-----------|
        | 2026-05-01 | 300.12 | 305.45 | 298.30 | 303.20 | 50000000  |
        | 2026-05-02 | 303.50 | 308.10 | 301.00 | 306.75 | 48000000  |
        | ...        | ...    | ...    | ...    | ...    | ...       |

        At least **60 rows** are required (matching `SEQ_LEN` used during training),
        with the most recent date as the **last row**. `Date` and `Volume`
        columns are optional but enable richer charts.
        """
    )
