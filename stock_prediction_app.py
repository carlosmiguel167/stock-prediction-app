# ================================================
#   STOCK PREDICTION APP — Streamlit Edition
#   Bold & Colorful | Built with Python & ML
# ================================================
 
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")
 
# ================================================
# PAGE CONFIG
# ================================================
 
st.set_page_config(
    page_title="StockSeer — AI Stock Predictor",
    page_icon="🔮",
    layout="wide"
)
 
# ================================================
# CUSTOM CSS — Bold & Colorful Theme
# ================================================
 
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');
 
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #f0f0f0;
  }
 
  .stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0a0f 100%);
  }
 
  h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
  }
 
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #FF6B6B, #FFD93D, #6BCB77, #4D96FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 0.2rem;
  }
 
  .hero-sub {
    font-size: 1.1rem;
    color: #888;
    margin-bottom: 2rem;
    font-family: 'DM Sans', sans-serif;
  }
 
  .prediction-card-up {
    background: linear-gradient(135deg, #1a3a1a, #0f2a0f);
    border: 2px solid #6BCB77;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(107, 203, 119, 0.2);
  }
 
  .prediction-card-down {
    background: linear-gradient(135deg, #3a1a1a, #2a0f0f);
    border: 2px solid #FF6B6B;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(255, 107, 107, 0.2);
  }
 
  .prediction-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
  }
 
  .prediction-direction {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    margin: 0.3rem 0;
  }
 
  .prediction-up   { color: #6BCB77; }
  .prediction-down { color: #FF6B6B; }
 
  .prediction-price {
    font-size: 1.4rem;
    color: #FFD93D;
    font-weight: 500;
    margin-top: 0.5rem;
  }
 
  .confidence-bar-wrap {
    background: #1a1a2e;
    border-radius: 50px;
    height: 12px;
    margin: 1rem 0 0.3rem 0;
    overflow: hidden;
  }
 
  .metric-card {
    background: linear-gradient(135deg, #12122a, #1a1a35);
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
 
  .metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #4D96FF;
  }
 
  .metric-label {
    font-size: 0.8rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.2rem;
  }
 
  .section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #FFD93D;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 2rem 0 1rem 0;
    border-left: 4px solid #FFD93D;
    padding-left: 0.8rem;
  }
 
  .stTextInput > div > div > input {
    background-color: #12122a !important;
    border: 2px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1.1rem !important;
    padding: 0.7rem 1rem !important;
  }
 
  .stTextInput > div > div > input:focus {
    border-color: #4D96FF !important;
    box-shadow: 0 0 0 2px rgba(77, 150, 255, 0.2) !important;
  }
 
  .stButton > button {
    background: linear-gradient(90deg, #FF6B6B, #FFD93D) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    letter-spacing: 1px !important;
    transition: all 0.2s ease !important;
  }
 
  .stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4) !important;
  }
 
  div[data-testid="stSelectbox"] > div {
    background-color: #12122a !important;
    border: 2px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #f0f0f0 !important;
  }
 
  .disclaimer {
    background: #12122a;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    font-size: 0.8rem;
    color: #555;
    margin-top: 3rem;
    text-align: center;
  }
 
  footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)
 
 
# ================================================
# ML FUNCTIONS
# ================================================
 
@st.cache_data(show_spinner=False)
def prepare_data(ticker_symbol, period="2y"):
    data = yf.Ticker(ticker_symbol).history(period=period)
    if data.empty:
        return None
    data = data[["Open", "High", "Low", "Close", "Volume"]]
    data["Daily_Return"]  = ((data["Close"] - data["Open"]) / data["Open"]) * 100
    data["MA20"]          = data["Close"].rolling(window=20).mean()
    data["MA50"]          = data["Close"].rolling(window=50).mean()
    data["Volatility"]    = data["High"] - data["Low"]
    data["Volume_Change"] = data["Volume"].pct_change() * 100
    data["MA_Signal"]     = (data["MA20"] > data["MA50"]).astype(int)
    data["Lag_1"]         = data["Close"].shift(1)
    data["Lag_2"]         = data["Close"].shift(2)
    data["Lag_3"]         = data["Close"].shift(3)
    data["Target"]        = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data = data.dropna()
    return data
 
@st.cache_data(show_spinner=False)
def run_model(ticker_symbol, period="2y"):
    data = prepare_data(ticker_symbol, period)
    if data is None:
        return None
 
    feature_cols = ["MA20", "MA50", "Daily_Return", "Volatility",
                    "MA_Signal", "Lag_1", "Lag_2", "Lag_3", "Volume_Change"]
 
    split   = int(len(data) * 0.80)
    train   = data.iloc[:split]
    test    = data.iloc[split:]
 
    model   = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(train[feature_cols], train["Target"])
 
    predictions = model.predict(test[feature_cols])
    accuracy    = accuracy_score(test["Target"], predictions)
 
    today       = data[feature_cols].iloc[[-1]]
    pred        = model.predict(today)[0]
    proba       = model.predict_proba(today)[0]
 
    results             = test[["Close"]].copy()
    results["Predicted"] = predictions
    results["Actual"]    = test["Target"].values
    results["Correct"]   = results["Predicted"] == results["Actual"]
 
    importances = dict(zip(feature_cols, model.feature_importances_))
 
    return {
        "data":        data,
        "test":        test,
        "results":     results,
        "accuracy":    accuracy,
        "pred":        pred,
        "proba":       proba,
        "importances": importances,
        "feature_cols": feature_cols,
    }
 
 
# ================================================
# CHART FUNCTIONS
# ================================================
 
def make_price_chart(data, ticker):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
 
    ax.plot(data.index, data["Close"], color="#4D96FF", linewidth=1.8, label="Close Price", zorder=3)
    ax.plot(data.index, data["MA20"],  color="#FFD93D", linewidth=1.2, linestyle="--", label="MA20", alpha=0.8)
    ax.plot(data.index, data["MA50"],  color="#FF6B6B", linewidth=1.2, linestyle="--", label="MA50", alpha=0.8)
    ax.fill_between(data.index,
                    data["Close"].min(), data["Close"].max(),
                    where=data["MA_Signal"] == 1,
                    alpha=0.06, color="#6BCB77", label="Bullish Zone")
 
    ax.set_title(f"{ticker} — Price History & Moving Averages",
                 color="#f0f0f0", fontsize=13, pad=12)
    ax.tick_params(colors="#666")
    ax.spines["bottom"].set_color("#2a2a4a")
    ax.spines["left"].set_color("#2a2a4a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.label.set_color("#666")
    ax.set_ylabel("Price (USD)", color="#666")
    ax.legend(facecolor="#12122a", edgecolor="#2a2a4a",
              labelcolor="#f0f0f0", fontsize=9)
    ax.grid(True, alpha=0.1, color="#4D96FF")
    plt.tight_layout()
    return fig
 
def make_prediction_chart(results, ticker):
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
 
    ax.plot(results.index, results["Close"],
            color="#888", linewidth=1.2, label="Close Price", zorder=2)
 
    correct   = results[results["Correct"] == True]
    incorrect = results[results["Correct"] == False]
 
    ax.scatter(correct.index,   correct["Close"],
               color="#6BCB77", s=30, label="Correct ✓", zorder=5, alpha=0.9)
    ax.scatter(incorrect.index, incorrect["Close"],
               color="#FF6B6B", s=30, label="Incorrect ✗", zorder=5, alpha=0.9)
 
    ax.set_title(f"{ticker} — Predictions vs Reality (Test Period)",
                 color="#f0f0f0", fontsize=13, pad=12)
    ax.tick_params(colors="#666")
    ax.spines["bottom"].set_color("#2a2a4a")
    ax.spines["left"].set_color("#2a2a4a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("Price (USD)", color="#666")
    ax.legend(facecolor="#12122a", edgecolor="#2a2a4a",
              labelcolor="#f0f0f0", fontsize=9)
    ax.grid(True, alpha=0.1, color="#4D96FF")
    plt.tight_layout()
    return fig
 
def make_importance_chart(importances):
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features     = [i[0] for i in sorted_items]
    values       = [i[1] * 100 for i in sorted_items]
 
    colors = ["#FF6B6B", "#FF8E53", "#FFD93D", "#C3FF6B",
              "#6BCB77", "#4DCFCB", "#4D96FF", "#9B6BFF", "#FF6BB5"]
 
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
 
    bars = ax.barh(features, values, color=colors[:len(features)],
                   height=0.6, edgecolor="none")
 
    for bar, val in zip(bars, values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", color="#f0f0f0", fontsize=9)
 
    ax.set_title("Feature Importance", color="#f0f0f0", fontsize=13, pad=12)
    ax.tick_params(colors="#888")
    ax.spines["bottom"].set_color("#2a2a4a")
    ax.spines["left"].set_color("#2a2a4a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Importance (%)", color="#666")
    ax.grid(True, alpha=0.1, axis="x", color="#4D96FF")
    plt.tight_layout()
    return fig
 
 
# ================================================
# APP LAYOUT
# ================================================
 
# --- Hero Header ---
st.markdown('<div class="hero-title">🔮 StockSeer</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">AI-powered stock direction predictor — built with Python & Machine Learning</div>', unsafe_allow_html=True)
 
# --- Sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")
 
    ticker_input = st.text_input(
        "Stock Ticker",
        value="AAPL",
        help="Type any ticker: AAPL, TSLA, MSFT, PETR4.SA"
    ).upper().strip()
 
    period_map = {
        "1 Year":  "1y",
        "2 Years": "2y",
        "5 Years": "5y"
    }
    period_label = st.selectbox("Historical Data", list(period_map.keys()), index=1)
    period       = period_map[period_label]
 
    st.markdown("---")
    run_button = st.button("🚀 Run Prediction")
 
    st.markdown("---")
    st.markdown("**Popular Tickers**")
    st.markdown("🇺🇸 `AAPL` `TSLA` `MSFT` `GOOGL` `NVDA`")
    st.markdown("🇧🇷 `PETR4.SA` `VALE3.SA` `ITUB4.SA`")
 
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#444;'>"
        "Built by a Python learner 🐍<br>"
        "Powered by yfinance + scikit-learn"
        "</div>",
        unsafe_allow_html=True
    )
 
 
# ================================================
# MAIN CONTENT
# ================================================
 
if run_button or "results" in st.session_state:
 
    if run_button:
        st.session_state["ticker"] = ticker_input
        st.session_state["period"] = period
 
    ticker = st.session_state.get("ticker", ticker_input)
    period = st.session_state.get("period", period)
 
    with st.spinner(f"🔄 Fetching {ticker} data and training model..."):
        out = run_model(ticker, period)
 
    if out is None:
        st.error(f"❌ Could not find data for **{ticker}**. Please check the ticker and try again.")
    else:
        data        = out["data"]
        results     = out["results"]
        accuracy    = out["accuracy"]
        pred        = out["pred"]
        proba       = out["proba"]
        importances = out["importances"]
 
        st.session_state["results"] = True
 
        # ---- PREDICTION CARD ----
        st.markdown('<div class="section-title">🔮 Tomorrow\'s Prediction</div>', unsafe_allow_html=True)
 
        card_class  = "prediction-card-up"   if pred == 1 else "prediction-card-down"
        dir_class   = "prediction-up"         if pred == 1 else "prediction-down"
        direction   = "📈 PRICE GOING UP"     if pred == 1 else "📉 PRICE GOING DOWN"
        confidence  = max(proba) * 100
        today_close = data["Close"].iloc[-1]
 
        st.markdown(f"""
        <div class="{card_class}">
            <div class="prediction-label">AI Prediction for {ticker}</div>
            <div class="prediction-direction {dir_class}">{direction}</div>
            <div class="prediction-price">Today's Close: ${today_close:.2f}</div>
            <div style="margin-top:1.2rem;">
                <div style="font-size:0.85rem; color:#888; margin-bottom:0.4rem;">
                    Confidence: {confidence:.1f}%
                </div>
                <div class="confidence-bar-wrap">
                    <div style="
                        height:100%;
                        width:{confidence}%;
                        background: linear-gradient(90deg, #4D96FF, {'#6BCB77' if pred==1 else '#FF6B6B'});
                        border-radius:50px;
                        transition: width 1s ease;">
                    </div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#666;">
                    <span>DOWN {proba[0]*100:.1f}%</span>
                    <span>UP {proba[1]*100:.1f}%</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        # ---- METRICS ROW ----
        st.markdown('<div class="section-title">📊 Model Stats</div>', unsafe_allow_html=True)
 
        total     = len(results)
        n_correct = results["Correct"].sum()
        up_days   = int(data["Target"].sum())
        down_days = len(data) - up_days
 
        col1, col2, col3, col4 = st.columns(4)
 
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{accuracy*100:.1f}%</div>
                <div class="metric-label">Model Accuracy</div>
            </div>""", unsafe_allow_html=True)
 
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{n_correct}/{total}</div>
                <div class="metric-label">Correct Predictions</div>
            </div>""", unsafe_allow_html=True)
 
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#6BCB77">{up_days}</div>
                <div class="metric-label">Up Days in Data</div>
            </div>""", unsafe_allow_html=True)
 
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color:#FF6B6B">{down_days}</div>
                <div class="metric-label">Down Days in Data</div>
            </div>""", unsafe_allow_html=True)
 
        st.markdown("<br>", unsafe_allow_html=True)
 
        # ---- CHARTS ----
        st.markdown('<div class="section-title">📈 Price History</div>', unsafe_allow_html=True)
        st.pyplot(make_price_chart(data, ticker))
 
        st.markdown('<div class="section-title">🎯 Prediction Accuracy</div>', unsafe_allow_html=True)
        st.pyplot(make_prediction_chart(results, ticker))
 
        col_left, col_right = st.columns([1, 1])
        with col_left:
            st.markdown('<div class="section-title">🧠 Feature Importance</div>', unsafe_allow_html=True)
            st.pyplot(make_importance_chart(importances))
 
        with col_right:
            st.markdown('<div class="section-title">📋 Classification Report</div>', unsafe_allow_html=True)
            report = classification_report(
                results["Actual"], results["Predicted"],
                target_names=["DOWN ↓", "UP ↑"],
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(
                report_df.style
                    .background_gradient(cmap="Blues", subset=["precision", "recall", "f1-score"])
                    .format(precision=2),
                use_container_width=True
            )
 
        # ---- DISCLAIMER ----
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This app is a learning project and is not financial advice.
            Stock predictions are inherently uncertain. Never make investment decisions based solely on ML model outputs.
        </div>
        """, unsafe_allow_html=True)
 
else:
    # Welcome state
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#444;">
        <div style="font-size:5rem; margin-bottom:1rem;">🔮</div>
        <div style="font-family:'Syne',sans-serif; font-size:1.5rem; color:#666; margin-bottom:0.5rem;">
            Enter a ticker and hit <strong style="color:#FFD93D">Run Prediction</strong>
        </div>
        <div style="font-size:0.9rem; color:#333;">
            Try AAPL, TSLA, MSFT, NVDA — or any Brazilian stock like PETR4.SA
        </div>
    </div>
    """, unsafe_allow_html=True)
