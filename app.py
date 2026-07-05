# ─────────────────────────────────────────────────────────────────────────────
#  India Monthly Temperature Forecasting Dashboard
#  Author : SEANG Vutha | ID: M080405
#  Course : Time Series Analysis and Forecasting — M-DAS / ITC
#  Run    : streamlit run app.py
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="India Temperature Forecasting",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f7f8fa;
    border-right: 1px solid #e8eaed;
}
[data-testid="stSidebar"] * { color: #2c3e50 !important; }

/* Hide default radio styling, style nav links */
[data-testid="stSidebar"] .stRadio > div { gap: 2px; }
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 0.88rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    display: block;
}
[data-testid="stSidebar"] .stRadio label:hover { background: #e8f4fd; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #eaf4fb 0%, #dbeeff 60%, #e8f5e9 100%);
    border-radius: 16px;
    padding: 2.5rem 2.8rem;
    margin-bottom: 1.8rem;
    border: 1px solid #c9e3f5;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #1a2e44;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    font-size: 0.97rem;
    color: #4a6070;
    margin: 0 0 1.2rem 0;
    max-width: 620px;
    line-height: 1.6;
}
.hero-tags { display: flex; gap: 8px; flex-wrap: wrap; }
.hero-tag {
    background: white;
    border: 1px solid #c5ddf0;
    color: #2980b9;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: white;
    border: 1px solid #e2eaf2;
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    min-height: 90px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.metric-card .mc-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #7f8c9a;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}
.metric-card .mc-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #1a2e44;
    line-height: 1.1;
}
.metric-card .mc-sub {
    font-size: 0.75rem;
    color: #95a5a6;
    margin-top: 4px;
}

/* Section title */
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1a2e44;
    margin: 1.4rem 0 0.8rem 0;
    padding-bottom: 6px;
    border-bottom: 2px solid #e8eaed;
}

/* Info cards (like the NLP card grid) */
.info-card {
    background: #f7fbff;
    border: 1px solid #d5e8f5;
    border-radius: 10px;
    padding: 1rem 1.1rem;
}
.info-card h4 {
    font-size: 0.85rem;
    font-weight: 700;
    color: #2980b9;
    margin: 0 0 6px 0;
}
.info-card p {
    font-size: 0.82rem;
    color: #4a6070;
    margin: 0;
    line-height: 1.5;
}

/* Model chip */
.model-chip {
    display: inline-block;
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    margin: 3px;
}

/* Page header (non-home pages) */
.page-header {
    background: linear-gradient(90deg, #eaf4fb, #f0f9ff);
    border-radius: 12px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.5rem;
    border-left: 5px solid #2980b9;
}
.page-header h2 {
    margin: 0;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1a2e44;
}
.page-header p {
    margin: 4px 0 0 0;
    font-size: 0.85rem;
    color: #5d7b8a;
}

/* Best model highlight cards */
.best-card {
    border-radius: 12px;
    padding: 1rem 1.2rem;
    border-left: 5px solid;
    margin-bottom: 12px;
}

/* Streamlit native metric override */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid #e2eaf2;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def metric_card(label, value, sub="", icon=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="mc-label">{icon} {label}</div>
        <div class="mc-value">{value}</div>
        {"<div class='mc-sub'>" + sub + "</div>" if sub else ""}
    </div>
    """, unsafe_allow_html=True)

def page_header(title, subtitle=""):
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner="Loading dataset…")
def load_data():
    import kagglehub
    path = kagglehub.dataset_download("venky73/temperatures-of-india")
    df_raw = pd.read_csv(path + "/temperatures.csv")
    df_melted = df_raw.melt(
        id_vars=["YEAR"],
        value_vars=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"],
        var_name="MONTH", value_name="TEMPERATURE",
    )
    month_order = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    df_melted["MONTH"] = pd.Categorical(df_melted["MONTH"], categories=month_order, ordered=True)
    df_melted = df_melted.sort_values(["YEAR","MONTH"]).reset_index(drop=True)
    df_melted["DATE"] = df_melted["MONTH"].astype(str) + "-" + df_melted["YEAR"].astype(str)
    df = df_melted[["DATE","TEMPERATURE"]].copy()
    df["DATE"] = pd.to_datetime(df["DATE"], format="%b-%Y")
    df.set_index("DATE", inplace=True)
    df.index.name = "Date"
    df = df.sort_index().asfreq("MS")
    df.dropna(inplace=True)
    return df, df_raw

df, df_raw = load_data()

train_size = int(len(df) * 0.8)
train = df["TEMPERATURE"][:train_size]
test  = df["TEMPERATURE"][train_size:]

def create_lag_features(series, n_lags=12):
    d = pd.DataFrame()
    for i in range(1, n_lags+1):
        d[f"lag_{i}"] = series.shift(i)
    d["target"] = series.values
    d.dropna(inplace=True)
    return d

df_lag = create_lag_features(df["TEMPERATURE"], n_lags=12)
X = df_lag.drop(columns="target"); y = df_lag["target"]
split_lag = int(len(X)*0.8)
X_train, X_test = X.iloc[:split_lag], X.iloc[split_lag:]
y_train, y_test = y.iloc[:split_lag], y.iloc[split_lag:]

t_all = np.arange(len(df)); s = 12
harmonics = pd.DataFrame(index=df.index)
harmonics["t"]    = t_all
harmonics["cos1"] = np.cos(2*np.pi*1*t_all/s)
harmonics["sin1"] = np.sin(2*np.pi*1*t_all/s)
harmonics["cos2"] = np.cos(2*np.pi*2*t_all/s)
harmonics["sin2"] = np.sin(2*np.pi*2*t_all/s)
train_harm = harmonics.iloc[:train_size]
test_harm  = harmonics.iloc[train_size:]
y_train_harm = df["TEMPERATURE"].iloc[:train_size]
y_test_harm  = df["TEMPERATURE"].iloc[train_size:]

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Fitting ARIMA…")
def fit_arima_model(_train):
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA
    auto_model = auto_arima(_train, start_p=0, max_p=5, start_q=0, max_q=5,
                            d=1, seasonal=False, stepwise=True,
                            error_action="ignore", suppress_warnings=True, trace=False)
    fit = ARIMA(_train, order=auto_model.order).fit()
    return fit, auto_model.order, auto_model

@st.cache_resource(show_spinner="Fitting Harmonic+ARIMA…")
def fit_harm_arima_model(_train):
    from pmdarima import auto_arima
    from sklearn.linear_model import LinearRegression
    ti = np.arange(len(_train)); p=12
    h_tr = pd.DataFrame({"sin":np.sin(2*np.pi*ti/p),"cos":np.cos(2*np.pi*ti/p)},index=_train.index)
    h1m = LinearRegression(); h1m.fit(h_tr, _train)
    hf  = pd.Series(h1m.predict(h_tr), index=_train.index)
    res = _train - hf
    am  = auto_arima(res, start_p=0, max_p=5, start_q=0, max_q=5,
                     d=0, seasonal=False, stepwise=True, trace=False,
                     error_action="ignore", suppress_warnings=True)
    return h1m, am, am.order

@st.cache_resource(show_spinner="Fitting SARIMA…")
def fit_sarima_model(_train):
    from pmdarima import auto_arima
    m = auto_arima(_train, start_p=0, max_p=3, start_q=0, max_q=3,
                   d=1, seasonal=True, m=12, start_P=0, max_P=2,
                   start_Q=0, max_Q=2, D=1, stepwise=True, trace=False,
                   error_action="ignore", suppress_warnings=True)
    return m, m.order, m.seasonal_order

@st.cache_resource(show_spinner="Fitting Harmonic Regression…")
def fit_harmonic_model(_train_harm, _y_train_harm):
    from sklearn.linear_model import LinearRegression
    m = LinearRegression(); m.fit(_train_harm, _y_train_harm)
    return m

@st.cache_resource(show_spinner="Fitting Random Forest…")
def fit_rf_model(_X_train, _y_train):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    m.fit(_X_train, _y_train); return m

@st.cache_resource(show_spinner="Fitting XGBoost…")
def fit_xgb_model(_X_train, _y_train, _X_test, _y_test):
    from xgboost import XGBRegressor
    m = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                     subsample=0.8, random_state=42)
    m.fit(_X_train, _y_train, eval_set=[(_X_test, _y_test)], verbose=False)
    return m

@st.cache_resource(show_spinner="Fitting Prophet…")
def fit_prophet_model(_train, _train_size):
    from prophet import Prophet
    df_p = _train.reset_index(); df_p.columns = ["ds","y"]
    m = Prophet(yearly_seasonality=True, changepoint_prior_scale=0.1, seasonality_mode="additive")
    m.fit(df_p); return m

def metrics(actual, predicted):
    a, p = np.array(actual), np.array(predicted)
    mae  = mean_absolute_error(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    mape = np.mean(np.abs((a-p)/a))*100
    return mae, rmse, mape

def verdict(gap):
    if gap < 0.3:   return "✅ No significant overfitting"
    elif gap < 0.8: return "⚠️ Mild overfitting"
    else:           return "🔴 Significant overfitting"

def diag_plot(resid, label):
    res = np.array(resid)
    std_res = (res - res.mean()) / res.std()
    (osm, osr), (sl, inc, _) = stats.probplot(std_res, dist="norm")
    from statsmodels.tsa.stattools import acf
    acf_v = acf(res, nlags=10)
    ci = 1.96 / np.sqrt(len(res))
    fig = make_subplots(rows=2, cols=2,
        subplot_titles=("Standardized Residuals","Histogram + N(0,1)","Normal Q-Q","Correlogram (ACF)"))
    fig.add_trace(go.Scatter(y=std_res, mode="lines", line=dict(color="#2980b9",width=1)), row=1, col=1)
    fig.add_hline(y=0, line_color="black", line_width=1, row=1, col=1)
    x_h = np.linspace(-4,4,100)
    fig.add_trace(go.Histogram(x=std_res, histnorm="probability density",
        marker_color="#2980b9", opacity=0.7), row=1, col=2)
    fig.add_trace(go.Scatter(x=x_h, y=stats.norm.pdf(x_h,0,1), mode="lines",
        line=dict(color="#27ae60",width=2)), row=1, col=2)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
        marker=dict(color="#2980b9",size=4)), row=2, col=1)
    fig.add_trace(go.Scatter(x=osm, y=sl*osm+inc, mode="lines",
        line=dict(color="#e74c3c",width=2)), row=2, col=1)
    fig.add_shape(type="rect", x0=-0.5,y0=-ci,x1=10.5,y1=ci,
        fillcolor="rgba(41,128,185,0.15)", line_width=0, row=2, col=2)
    for l, v in enumerate(acf_v):
        fig.add_shape(type="line",x0=l,y0=0,x1=l,y1=v,
            line=dict(color="#2980b9",width=2), row=2, col=2)
    fig.add_trace(go.Scatter(x=np.arange(11), y=acf_v, mode="markers",
        marker=dict(color="#2980b9",size=6)), row=2, col=2)
    fig.add_shape(type="line",x0=-0.5,y0=0,x1=10.5,y1=0,
        line=dict(color="black",width=1), row=2, col=2)
    fig.update_layout(height=560, template="plotly_white", showlegend=False,
        title={"text":label,"x":0.5}, paper_bgcolor="white", plot_bgcolor="#fafcff")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0.5rem 0.8rem 0.5rem;'>
        <div style='font-size:1.15rem;font-weight:800;color:#1a2e44;'>🌡️ India Temperature</div>
        <div style='font-size:0.78rem;color:#7f8c9a;margin-top:2px;'>Forecasting Dashboard</div>
    </div>
    <hr style='margin:0 0 0.8rem 0;border:none;border-top:1px solid #e8eaed;'>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠 Home",
        "⛁ Dataset",
        "📊 EDA",
        "📈 Statistical Analysis",
        "🤖 Forecasting Models",
        "📉 Residual Diagnostics",
        "🏆 Model Comparison",
        "🔮 Future Forecast",
        "ℹ️ About",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='margin:0.8rem 0;border:none;border-top:1px solid #e8eaed;'>
    <div style='font-size:0.72rem;color:#aab;padding:0 0.5rem;line-height:1.6;'>
        <b>Course:</b> Time Series Analysis<br>
        <b>Author:</b> SEANG Vutha<br>
        <b>ID:</b> M080405<br>
        <b>Supervisor:</b> Dr. NHIM Malai
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    # Hero Banner
    st.markdown("""
    <div class="hero-banner">
        <h1>🌡️ India Monthly Temperature Forecasting</h1>
        <p>Comprehensive forecasting study on monthly average temperature across India (1901–2017) —
        117 years of observations capturing monsoon cycles, long-term warming trends, and climate change signals.</p>
        <div class="hero-tags">
            <span class="hero-tag">📊 Classical Models</span>
            <span class="hero-tag">🤖 Machine Learning</span>
            <span class="hero-tag">1,404 Records</span>
            <span class="hero-tag">117 Years</span>
            <span class="hero-tag">7 Models</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metric cards row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Records", f"{len(df):,}", "Monthly observations", "📅")
    with c2:
        metric_card("Date Range", "1901 – 2017", "117 years of data", "📆")
    with c3:
        metric_card("Mean Temp", f"{df['TEMPERATURE'].mean():.1f} °C", "Monthly average", "🌡️")
    with c4:
        metric_card("Warming", "+1.5 °C", "Over 117 years", "📈")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9])
    with col1:
        section_title("📌 Project Description")
        st.markdown("""
        This study implements **7 forecasting models** from two paradigms to analyze and predict
        India's monthly temperature across a 117-year record (1901–2017).
        """)
        st.markdown("""
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:10px;'>
            <div class="info-card">
                <h4>Classical Models</h4>
                <p>ARIMA · SARIMA · Harmonic Regression · Harmonic+ARIMA hybrid</p>
            </div>
            <div class="info-card">
                <h4>Machine Learning</h4>
                <p>Random Forest · XGBoost · Prophet</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        section_title("🎯 Objectives")
        objectives = [
            "EDA on India's monthly temperature dataset",
            "Stationarity testing (ADF) and differencing",
            "ACF/PACF analysis for ARIMA order selection",
            "Seasonal decomposition (additive model)",
            "Implement and compare all 7 forecasting models",
            "Evaluate with MAE, RMSE, MAPE metrics",
            "Forecast future temperatures beyond December 2017",
        ]
        for obj in objectives:
            st.markdown(f"<div style='font-size:0.85rem;padding:3px 0;color:#2c3e50;'>▸ {obj}</div>", unsafe_allow_html=True)

    with col2:
        section_title("📊 Dataset Summary")
        desc = df["TEMPERATURE"].describe().round(2)
        stats_df = pd.DataFrame({
            "Statistic": ["Count","Mean","Std Dev","Min","25th Pct","Median","75th Pct","Max"],
            "Value": [
                f"{int(desc['count']):,}",
                f"{desc['mean']} °C",
                f"{desc['std']} °C",
                f"{desc['min']} °C",
                f"{desc['25%']} °C",
                f"{desc['50%']} °C",
                f"{desc['75%']} °C",
                f"{desc['max']} °C",
            ]
        })
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

        section_title("🤖 Models Used")
        model_info = [
            ("ARIMA",             "#e74c3c"),
            ("SARIMA",            "#8e44ad"),
            ("Harmonic Reg",      "#2980b9"),
            ("Harmonic+ARIMA",    "#16a085"),
            ("Random Forest",     "#27ae60"),
            ("XGBoost",           "#f39c12"),
            ("Prophet",           "#c0392b"),
        ]
        chips = "".join([
            f"<span class='model-chip' style='background:{c};'>{n}</span>"
            for n, c in model_info
        ])
        st.markdown(f"<div style='margin-top:6px;'>{chips}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⛁ Dataset":
    page_header("⛁ Dataset", "India monthly temperature data (1901–2017) — Kaggle source")

    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Rows", f"{df.shape[0]:,}", "After transformation", "📋")
    with c2: metric_card("Columns", str(df.shape[1]), "Temperature only", "📐")
    with c3: metric_card("Missing Values", "0", "Complete record", "✅")

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📋 Overview", "🔢 Statistics", "📥 Download"])

    with t1:
        section_title("Processed Dataset (Long Format)")
        st.dataframe(df, use_container_width=True, height=400)
        section_title("Raw Dataset (Wide Format)")
        st.dataframe(df_raw, use_container_width=True, height=400)
        with st.expander("📖 Transformation Steps"):
            steps = [
                ("Column Cleanup", "Removed ANNUAL and JAN–FEB aggregated columns"),
                ("Data Melting", "Reshaped 12 monthly columns into one TEMPERATURE column using pandas.melt()"),
                ("Categorical Ordering", "Months encoded as ordered Categorical (Jan–Dec)"),
                ("Datetime Indexing", "Combined year+month into DatetimeIndex with monthly-start frequency (MS)"),
            ]
            for i, (title, desc) in enumerate(steps, 1):
                st.markdown(f"**{i}. {title}** — {desc}")

    with t2:
        st.dataframe(df.describe().round(4), use_container_width=True)

    with t3:
        st.download_button("📥 Download Processed CSV", df.to_csv().encode(), "india_temperature.csv", "text/csv")
        st.download_button("📥 Download Raw CSV", df_raw.to_csv(index=False).encode(), "india_temperature_raw.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    page_header("📊 Exploratory Data Analysis", "Visual analysis of India's 117-year temperature record")

    t1, t2, t3, t4 = st.tabs(["📈 Time Series", "🔄 Decomposition", "📉 Rolling Stats", "📅 Monthly Patterns"])

    with t1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["TEMPERATURE"], mode="lines",
            line=dict(color="#e74c3c",width=1), fill="tozeroy", fillcolor="rgba(231,76,60,0.07)"))
        fig.update_layout(title={"text":"India Monthly Temperature (1901–2017)","x":0.5},
            xaxis_title="Date", yaxis_title="Temperature (°C)",
            template="plotly_white", paper_bgcolor="white", plot_bgcolor="#fafcff", height=420)
        fig.update_xaxes(showgrid=True, gridcolor="#eaecee", rangeslider_visible=True,
            rangeselector=dict(buttons=[
                dict(count=10,label="10y",step="year",stepmode="backward"),
                dict(count=30,label="30y",step="year",stepmode="backward"),
                dict(count=60,label="60y",step="year",stepmode="backward"),
                dict(step="all",label="All")]))
        fig.update_yaxes(showgrid=True, gridcolor="#eaecee")
        st.plotly_chart(fig, use_container_width=True)
        findings = [
            ("Strong Seasonality", "Regular 12-month oscillation driven by India's monsoon cycle"),
            ("Long-term Warming Trend", "~+1.5°C drift from 1901 to 2017 (South Asian climate warming)"),
            ("Stable Variance", "Constant amplitude → additive decomposition is appropriate"),
        ]
        cols = st.columns(3)
        for col, (title, desc) in zip(cols, findings):
            col.markdown(f"""
            <div class="info-card">
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with t2:
        from statsmodels.tsa.seasonal import seasonal_decompose
        with st.spinner("Decomposing…"):
            decomp = seasonal_decompose(df["TEMPERATURE"], model="additive", period=12)
        components = {"Observed":decomp.observed,"Trend":decomp.trend,"Seasonal":decomp.seasonal,"Residual":decomp.resid}
        fig2 = make_subplots(rows=4, cols=1, subplot_titles=list(components.keys()), vertical_spacing=0.07)
        colors = ["#e74c3c","#2980b9","#8e44ad","#27ae60"]
        for i,((nm,data),clr) in enumerate(zip(components.items(),colors),1):
            fig2.add_trace(go.Scatter(x=data.index,y=data.values,mode="lines",
                line=dict(color=clr,width=1.3)), row=i, col=1)
        fig2.update_layout(title={"text":"Seasonal Decomposition (period=12)","x":0.5},
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",showlegend=False,height=800)
        st.plotly_chart(fig2, use_container_width=True)

    with t3:
        r12=df["TEMPERATURE"].rolling(12).mean()
        r60=df["TEMPERATURE"].rolling(60).mean()
        r120=df["TEMPERATURE"].rolling(120).mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=df.index,y=df["TEMPERATURE"],name="Original",
            line=dict(color="#bdc3c7",width=0.8),opacity=0.6))
        fig3.add_trace(go.Scatter(x=df.index,y=r12, name="12-month MA",line=dict(color="#8e44ad",width=2)))
        fig3.add_trace(go.Scatter(x=df.index,y=r60, name="5-year MA",  line=dict(color="#2980b9",width=2.5)))
        fig3.add_trace(go.Scatter(x=df.index,y=r120,name="10-year MA", line=dict(color="#27ae60",width=2.5)))
        fig3.update_layout(title={"text":"Temperature with Rolling Mean (12m, 5y, 10y)","x":0.5},
            xaxis_title="Date",yaxis_title="Temperature (°C)",template="plotly_white",
            paper_bgcolor="white",plot_bgcolor="#fafcff",hovermode="x unified",height=420,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig3, use_container_width=True)

    with t4:
        mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        mavg = df.groupby(df.index.month)["TEMPERATURE"].mean()
        fig4 = go.Figure(go.Bar(x=mnames, y=mavg.values,
            marker_color=["#e74c3c" if v>=mavg.mean() else "#2980b9" for v in mavg.values],
            text=mavg.round(1).values, textposition="outside"))
        fig4.update_layout(title={"text":"Monthly Average Temperature","x":0.5},
            xaxis_title="Month",yaxis_title="Avg Temperature (°C)",
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",height=400)
        st.plotly_chart(fig4, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Statistical Analysis":
    page_header("📈 Statistical Analysis", "Stationarity testing, ACF/PACF, and differencing")
    from statsmodels.tsa.stattools import adfuller, acf, pacf

    diff_series = df["TEMPERATURE"].diff().dropna()
    t1, t2, t3 = st.tabs(["🧪 Stationarity Tests", "📊 ACF & PACF", "📉 Differencing"])

    with t1:
        col1, col2 = st.columns(2)
        with col1:
            section_title("Kruskal–Wallis Test")
            groups=[df["TEMPERATURE"][df.index.month==m] for m in range(1,13)]
            kw_stat,kw_p=stats.kruskal(*groups)
            st.dataframe(pd.DataFrame({
                "Metric":["Test Statistic","p-value","Result"],
                "Value":[f"{kw_stat:.4f}",f"{kw_p:.2e}","✅ Reject H₀ — Seasonality confirmed"]
            }), use_container_width=True, hide_index=True)
            st.info("p < 0.001 → Statistically significant seasonal variation across all 12 months.")
        with col2:
            section_title("ADF Test")
            adf_o=adfuller(df["TEMPERATURE"].dropna())
            adf_d=adfuller(diff_series)
            st.dataframe(pd.DataFrame({
                "Test":["Original","After 1st Differencing"],
                "ADF Stat":[f"{adf_o[0]:.4f}",f"{adf_d[0]:.4f}"],
                "p-value":[f"{adf_o[1]:.4f}",f"{adf_d[1]:.4f}"],
                "Crit (5%)":[f"{adf_o[4]['5%']:.4f}",f"{adf_d[4]['5%']:.4f}"],
                "Result":["❌ Non-Stationary","✅ Stationary"]
            }), use_container_width=True, hide_index=True)
            st.info("After first-order differencing (d=1): ADF = −11.55, p ≈ 0 → Stationary.")

    with t2:
        acf_v=acf(diff_series,nlags=36); pacf_v=pacf(diff_series,nlags=36)
        lags=list(range(len(acf_v))); n=len(diff_series); ci=1.96/(n**0.5)
        fig=make_subplots(rows=2,cols=1,
            subplot_titles=("ACF — Temperature (After Differencing)","PACF — Temperature (After Differencing)"),
            vertical_spacing=0.15)
        fig.add_trace(go.Bar(x=lags,y=acf_v,name="ACF",width=0.6,
            marker=dict(color=["#27ae60" if abs(v)>ci else "#e74c3c" for v in acf_v],line=dict(width=0))),row=1,col=1)
        fig.add_trace(go.Bar(x=lags,y=pacf_v,name="PACF",width=0.6,
            marker=dict(color=["#27ae60" if abs(v)>ci else "#e74c3c" for v in pacf_v],line=dict(width=0))),row=2,col=1)
        for row in [1,2]:
            for yv in [ci,-ci]:
                fig.add_hline(y=yv,line_dash="dash",line_color="#95a5a6",line_width=1.3,opacity=0.7,row=row,col=1)
            fig.add_hline(y=0,line_color="black",line_width=1,row=row,col=1)
        fig.update_layout(title={"text":"ACF & PACF — Temperature (After Differencing)","x":0.5},
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",showlegend=False,height=600)
        st.plotly_chart(fig, use_container_width=True)
        c1,c2=st.columns(2)
        c1.markdown("""<div class="info-card"><h4>ACF Interpretation</h4>
        <p>Spikes at lags 12, 24, 36 → strong 12-month seasonal pattern. Between seasonal lags, tails off → MA component.</p>
        </div>""", unsafe_allow_html=True)
        c2.markdown("""<div class="info-card"><h4>PACF Interpretation</h4>
        <p>Significant spikes at lags 1–2 then sharp cutoff → AR(1) or AR(2). Suggests p=2, d=1 as starting point.</p>
        </div>""", unsafe_allow_html=True)

    with t3:
        fig2=go.Figure()
        fig2.add_trace(go.Scatter(x=diff_series.index,y=diff_series.values,mode="lines",
            line=dict(color="#27ae60",width=1.3)))
        fig2.add_hline(y=0,line_color="black",line_width=1)
        fig2.update_layout(title={"text":"Temperature — First Order Differencing","x":0.5},
            xaxis_title="Date",yaxis_title="Differenced Value (°C)",
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",height=380)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(pd.DataFrame({
            "Feature":["Mean","Variance","Trend","Stationarity"],
            "Original":["Increasing","Stable","Upward","❌ NO"],
            "After Differencing":["Constant ~0","Stable","Removed","✅ YES"]
        }), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FORECASTING MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Forecasting Models":
    page_header("🤖 Forecasting Models", "Select a model to view its forecast and evaluation metrics")

    MODEL_DESCS = {
        "ARIMA":              ("Classical", "#e74c3c", "ARIMA(p,d,q) — auto_arima found ARIMA(2,1,2). Without seasonal components, multi-step forecasts degrade to a flat line since the model cannot reproduce the 12-month cycle."),
        "Harmonic + ARIMA":   ("Classical", "#16a085", "Two-stage hybrid: Harmonic Regression captures the deterministic annual cycle, then ARIMA(3,0,3) models remaining stochastic residuals. Final forecast = harmonic + ARIMA residual."),
        "SARIMA":             ("Classical", "#8e44ad", "SARIMA(3,1,0)×(2,1,0)₁₂ — explicit seasonal AR/MA at s=12. Successfully reproduces annual peaks and valleys over the full 23-year test horizon."),
        "Harmonic Regression":("Classical", "#2980b9", "Deterministic model with two sin/cos pairs + linear trend. Only 5 parameters. Achieves best generalization stability (gap +0.36°C)."),
        "Random Forest":      ("ML",        "#27ae60", "Ensemble of 200 decision trees on 12 lag features. lag_12 accounts for 92.7% of feature importance. Boundary-level overfitting (gap +0.75°C)."),
        "XGBoost":            ("ML",        "#f39c12", "Sequential gradient boosting with built-in regularization. Best overall accuracy (RMSE 0.9453°C, MAPE 2.39%) across all 7 models."),
        "Prophet":            ("ML",        "#c0392b", "Facebook additive model: y(t)=g(t)+s(t)+ε. Handles long-term trend and 12-month seasonality automatically. changepoint_prior_scale=0.1."),
    }

    # Model selector as styled buttons
    model_name = st.selectbox("Select Model:", list(MODEL_DESCS.keys()))
    mtype, mcolor, mdesc = MODEL_DESCS[model_name]

    st.markdown(f"""
    <div style='background:#f7fbff;border:1px solid #d5e8f5;border-left:5px solid {mcolor};
    border-radius:10px;padding:1rem 1.2rem;margin:0.5rem 0 1rem 0;'>
        <span style='background:{mcolor};color:white;padding:2px 10px;border-radius:12px;
        font-size:0.75rem;font-weight:700;'>{mtype}</span>
        <p style='margin:8px 0 0 0;font-size:0.88rem;color:#2c3e50;line-height:1.6;'>{mdesc}</p>
    </div>
    """, unsafe_allow_html=True)

    def forecast_plot(tr, te, fc, name, color="#e74c3c"):
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=tr.index,y=tr,name="Train",line=dict(color="#2980b9",width=1.3)))
        fig.add_trace(go.Scatter(x=te.index,y=te,name="Actual",line=dict(color="#f39c12",width=2)))
        fig.add_trace(go.Scatter(x=fc.index,y=fc,name=f"{name} Forecast",
            line=dict(color=color,dash="dash",width=2)))
        fig.update_layout(title={"text":f"{name} Forecast vs Actual","x":0.5},
            xaxis_title="Date",yaxis_title="Temperature (°C)",
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",
            hovermode="x unified",height=420,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        return fig

    def show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr):
        c1,c2,c3,c4 = st.columns(4)
        with c1: metric_card("MAE", f"{mae:.4f} °C", f"Train: {mae_tr:.4f}", "📏")
        with c2: metric_card("RMSE", f"{rmse:.4f} °C", f"Train: {rmse_tr:.4f}", "📐")
        with c3: metric_card("MAPE", f"{mape:.2f} %", "Percentage error", "📊")
        gap=rmse-rmse_tr
        with c4: metric_card("Overfitting Gap", f"{gap:+.4f} °C", verdict(gap), "🔍")

    if model_name == "ARIMA":
        arima_fit, arima_order, auto_model = fit_arima_model(train)
        fc=arima_fit.forecast(steps=len(test)); fc.index=test.index
        mae,rmse,mape=metrics(test,fc)
        tp=arima_fit.fittedvalues.dropna()
        mae_tr,rmse_tr,_=metrics(train.loc[tp.index],tp)
        st.plotly_chart(forecast_plot(train,test,fc,"ARIMA"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)

    elif model_name == "Harmonic + ARIMA":
        h1m, am, order = fit_harm_arima_model(train)
        ti=np.arange(len(train)); p=12
        h_tr=pd.DataFrame({"sin":np.sin(2*np.pi*ti/p),"cos":np.cos(2*np.pi*ti/p)},index=train.index)
        hf=pd.Series(h1m.predict(h_tr),index=train.index)
        ti_te=np.arange(len(train),len(train)+len(test))
        h_te=pd.DataFrame({"sin":np.sin(2*np.pi*ti_te/p),"cos":np.cos(2*np.pi*ti_te/p)},index=test.index)
        fc_h=h1m.predict(h_te); fc_r=am.predict(n_periods=len(test))
        fc=pd.Series(fc_h+fc_r,index=test.index)
        mae,rmse,mape=metrics(test,fc)
        tp=hf+pd.Series(am.fittedvalues(),index=train.index)
        tp=tp.dropna(); mae_tr,rmse_tr,_=metrics(train.loc[tp.index],tp)
        st.plotly_chart(forecast_plot(train,test,fc,"Harmonic + ARIMA","#16a085"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)

    elif model_name == "SARIMA":
        sarima_m, sorder, ssorder = fit_sarima_model(train)
        fc_v=sarima_m.predict(n_periods=len(test)); fc=pd.Series(fc_v,index=test.index)
        mae,rmse,mape=metrics(test,fc)
        tp=pd.Series(sarima_m.fittedvalues()[5:],index=train.index[5:]).dropna()
        mae_tr,rmse_tr,_=metrics(train.loc[tp.index],tp)
        st.plotly_chart(forecast_plot(train,test,fc,"SARIMA","#8e44ad"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)

    elif model_name == "Harmonic Regression":
        hm = fit_harmonic_model(train_harm, y_train_harm)
        fc=pd.Series(hm.predict(test_harm),index=y_test_harm.index)
        mae,rmse,mape=metrics(y_test_harm,fc)
        tp=pd.Series(hm.predict(train_harm),index=y_train_harm.index)
        mae_tr,rmse_tr,_=metrics(y_train_harm,tp)
        st.plotly_chart(forecast_plot(train,test,fc,"Harmonic Regression","#2980b9"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)
        section_title("Fitted Coefficients")
        st.dataframe(pd.DataFrame({
            "Coefficient":["Intercept (β0)","Trend (t)","cos1 (β1)","sin1 (β2)","cos2 (β3)","sin2 (β4)"],
            "Value":[round(hm.intercept_,4)]+[round(c,6) for c in hm.coef_]
        }), use_container_width=True, hide_index=True)

    elif model_name == "Random Forest":
        rfm = fit_rf_model(X_train, y_train)
        fc_v=rfm.predict(X_test); fc=pd.Series(fc_v,index=y_test.index)
        mae,rmse,mape=metrics(y_test,fc)
        tp=pd.Series(rfm.predict(X_train),index=y_train.index)
        mae_tr,rmse_tr,_=metrics(y_train,tp)
        st.plotly_chart(forecast_plot(train,test,fc,"Random Forest","#27ae60"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)
        imp=pd.Series(rfm.feature_importances_,index=X_train.columns).sort_values(ascending=False)
        section_title("Top 5 Feature Importances")
        imp_df = imp.head(5).reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Importance %"] = (imp_df["Importance"]*100).round(2)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    elif model_name == "XGBoost":
        xgbm = fit_xgb_model(X_train,y_train,X_test,y_test)
        fc_v=xgbm.predict(X_test); fc=pd.Series(fc_v,index=y_test.index)
        mae,rmse,mape=metrics(y_test,fc)
        tp=pd.Series(xgbm.predict(X_train),index=y_train.index)
        mae_tr,rmse_tr,_=metrics(y_train,tp)
        st.plotly_chart(forecast_plot(train,test,fc,"XGBoost","#f39c12"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)
        imp=pd.Series(xgbm.feature_importances_,index=X_train.columns).sort_values(ascending=False)
        section_title("Top 5 Feature Importances")
        imp_df = imp.head(5).reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Importance %"] = (imp_df["Importance"]*100).round(2)
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

    elif model_name == "Prophet":
        pm = fit_prophet_model(train, train_size)
        df_p=train.reset_index(); df_p.columns=["ds","y"]
        future=pm.make_future_dataframe(periods=len(test),freq="MS")
        fc_full=pm.predict(future)
        fc=pd.Series(fc_full["yhat"].tail(len(test)).values,index=test.index)
        mae,rmse,mape=metrics(test,fc)
        tr_fc=pm.predict(df_p); tp=pd.Series(tr_fc["yhat"].values,index=train.index)
        mae_tr,rmse_tr,_=metrics(train,tp)
        st.plotly_chart(forecast_plot(train,test,fc,"Prophet","#c0392b"), use_container_width=True)
        show_metrics_row(mae,rmse,mape,mae_tr,rmse_tr)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESIDUAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📉 Residual Diagnostics":
    page_header("📉 Residual Diagnostics", "4-panel diagnostics for stochastic models: ARIMA, Harmonic+ARIMA, SARIMA")

    diag_choice = st.selectbox("Select Model:", [
        "ARIMA(2,1,2)", "Harmonic + ARIMA(3,0,3)", "SARIMA(3,1,0)×(2,1,0)₁₂"])

    from statsmodels.stats.diagnostic import acorr_ljungbox

    if diag_choice == "ARIMA(2,1,2)":
        fit, order, am = fit_arima_model(train)
        resid = fit.resid[1:]; lb_resid = fit.resid
        label = f"ARIMA{order} — Model Diagnostics"
        interp = {
            "Std Res":    "Fluctuate randomly around zero ✓ — no obvious pattern.",
            "Histogram":  "Bell-shaped and centred near zero, consistent with normality ✓.",
            "Q-Q":        "Points follow reference line well in middle; slight upper-tail deviation — mild, common in climate data.",
            "Correlogram":"Spikes at lags 5–7 reflect unmodelled 12-month cycle — expected for non-seasonal ARIMA.",
            "Overall":    "Normality confirmed ✓. Correlogram spikes at lags 5–7 are a known limitation — they motivate SARIMA and Harmonic Regression.",
        }
    elif diag_choice == "Harmonic + ARIMA(3,0,3)":
        h1m, am, order = fit_harm_arima_model(train)
        resid = am.resid()[5:]; lb_resid = am.resid()
        label = f"ARIMA{order} on Harmonic Residuals Diagnostics"
        interp = {
            "Std Res":    "Random around zero ✓ — slightly more variation post-1960 but nothing serious.",
            "Histogram":  "Close to N(0,1) but slightly right-skewed — broadly acceptable.",
            "Q-Q":        "Good fit in the middle; slight deviation at both tails — mild non-normality typical in climate residuals.",
            "Correlogram":"Only lag 1 slightly exceeds the boundary; all others well within bounds ✓.",
            "Overall":    "ARIMA(3,0,3) on harmonic residuals shows minimal autocorrelation — confirming the hybrid successfully separated deterministic seasonality from stochastic structure.",
        }
    else:
        sarima_m, sorder, ssorder = fit_sarima_model(train)
        resid = sarima_m.resid()[5:]; lb_resid = sarima_m.resid()
        label = f"SARIMA{sorder}×{ssorder} — Model Diagnostics"
        interp = {
            "Std Res":    "Mostly random around zero ✓. Large spikes in early period — likely unusual climate years, not model failure.",
            "Histogram":  "Centred at zero; slight left-skew due to early outliers — not a serious concern.",
            "Q-Q":        "Good fit across middle and upper range. Lower tail deviations driven by early outliers.",
            "Correlogram":"All lags within confidence bounds ✓ — no remaining autocorrelation. Seasonal pattern fully captured.",
            "Overall":    "SARIMA passes the correlogram check perfectly. Tail deviations are caused by a small number of extreme early-period observations, not systematic model failure.",
        }

    st.plotly_chart(diag_plot(resid, label), use_container_width=True)

    section_title("📝 Panel Interpretations")
    c1,c2 = st.columns(2)
    panels = [("Standardized Residuals", interp["Std Res"]),
              ("Normal Q-Q Plot", interp["Q-Q"]),
              ("Histogram + N(0,1)", interp["Histogram"]),
              ("Correlogram (ACF)", interp["Correlogram"])]
    for i, (title, desc) in enumerate(panels):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""<div class="info-card" style="margin-bottom:10px;">
        <h4>{title}</h4><p>{desc}</p></div>""", unsafe_allow_html=True)
    st.success(f"**Overall:** {interp['Overall']}")

    section_title("🧪 Ljung–Box Test")
    st.caption("H₀: residuals are white noise (p > 0.05 = pass)")
    try:
        lb = acorr_ljungbox(lb_resid, lags=[12,24], return_df=True)
        lb.index=["Lag 12","Lag 24"]; lb.columns=["LB Statistic","p-value"]
        lb["Result"]=lb["p-value"].apply(lambda p: "✅ White noise" if p>0.05 else "⚠️ Autocorrelation")
        st.dataframe(lb.round(4), use_container_width=True)
        if (lb["p-value"]>0.05).all():
            st.success("→ All p-values > 0.05: residuals are white noise ✅")
        else:
            st.warning("→ Some p-values ≤ 0.05: residual autocorrelation detected.")
    except Exception as e:
        st.warning(f"Ljung-Box could not be computed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    page_header("🏆 Model Comparison", "All 7 models ranked by accuracy and generalization")

    with st.spinner("Computing all model metrics…"):
        arima_fit, arima_order, _ = fit_arima_model(train)
        fc_arima=arima_fit.forecast(steps=len(test)); fc_arima.index=test.index
        mae_arima,rmse_arima,mape_arima=metrics(test,fc_arima)
        tp=arima_fit.fittedvalues.dropna(); rmse_arima_train=np.sqrt(mean_squared_error(train.loc[tp.index],tp))

        h1m,am,_=fit_harm_arima_model(train)
        ti=np.arange(len(train)); p=12
        h_tr=pd.DataFrame({"sin":np.sin(2*np.pi*ti/p),"cos":np.cos(2*np.pi*ti/p)},index=train.index)
        hf_=pd.Series(h1m.predict(h_tr),index=train.index)
        ti_te=np.arange(len(train),len(train)+len(test))
        h_te=pd.DataFrame({"sin":np.sin(2*np.pi*ti_te/p),"cos":np.cos(2*np.pi*ti_te/p)},index=test.index)
        fc_ha=pd.Series(h1m.predict(h_te)+am.predict(n_periods=len(test)),index=test.index)
        mae_ha,rmse_ha,mape_ha=metrics(test,fc_ha)
        tp_ha=(hf_+pd.Series(am.fittedvalues(),index=train.index)).dropna()
        rmse_ha_train=np.sqrt(mean_squared_error(train.loc[tp_ha.index],tp_ha))

        sarima_m,_,__=fit_sarima_model(train)
        fc_s=pd.Series(sarima_m.predict(n_periods=len(test)),index=test.index)
        mae_s,rmse_s,mape_s=metrics(test,fc_s)
        tp_s=pd.Series(sarima_m.fittedvalues()[5:],index=train.index[5:]).dropna()
        rmse_s_train=np.sqrt(mean_squared_error(train.loc[tp_s.index],tp_s))

        hm=fit_harmonic_model(train_harm,y_train_harm)
        fc_h=pd.Series(hm.predict(test_harm),index=y_test_harm.index)
        mae_h,rmse_h,mape_h=metrics(y_test_harm,fc_h)
        tp_h=pd.Series(hm.predict(train_harm),index=y_train_harm.index)
        rmse_h_train=np.sqrt(mean_squared_error(y_train_harm,tp_h))

        rfm=fit_rf_model(X_train,y_train)
        fc_rf=pd.Series(rfm.predict(X_test),index=y_test.index)
        mae_rf,rmse_rf,mape_rf=metrics(y_test,fc_rf)
        rmse_rf_train=np.sqrt(mean_squared_error(y_train,rfm.predict(X_train)))

        xgbm=fit_xgb_model(X_train,y_train,X_test,y_test)
        fc_xgb=pd.Series(xgbm.predict(X_test),index=y_test.index)
        mae_xgb,rmse_xgb,mape_xgb=metrics(y_test,fc_xgb)
        rmse_xgb_train=np.sqrt(mean_squared_error(y_train,xgbm.predict(X_train)))

        pm=fit_prophet_model(train,train_size)
        df_p=train.reset_index(); df_p.columns=["ds","y"]
        fut=pm.make_future_dataframe(periods=len(test),freq="MS")
        fc_prop=pd.Series(pm.predict(fut)["yhat"].tail(len(test)).values,index=test.index)
        mae_prop,rmse_prop,mape_prop=metrics(test,fc_prop)
        tp_prop=pd.Series(pm.predict(df_p)["yhat"].values,index=train.index)
        rmse_prop_train=np.sqrt(mean_squared_error(train,tp_prop))

    comp=pd.DataFrame({
        "Model":["XGBoost","Harmonic Regression","Prophet","Random Forest","Harmonic + ARIMA","SARIMA","ARIMA"],
        "Type":["ML","Classical","ML","ML","Classical","Classical","Classical"],
        "MAE (°C)":[round(mae_xgb,4),round(mae_h,4),round(mae_prop,4),round(mae_rf,4),round(mae_ha,4),round(mae_s,4),round(mae_arima,4)],
        "RMSE (°C)":[round(rmse_xgb,4),round(rmse_h,4),round(rmse_prop,4),round(rmse_rf,4),round(rmse_ha,4),round(rmse_s,4),round(rmse_arima,4)],
        "MAPE (%)":[round(mape_xgb,2),round(mape_h,2),round(mape_prop,2),round(mape_rf,2),round(mape_ha,2),round(mape_s,2),round(mape_arima,2)],
        "Train RMSE":[round(rmse_xgb_train,4),round(rmse_h_train,4),round(rmse_prop_train,4),round(rmse_rf_train,4),round(rmse_ha_train,4),round(rmse_s_train,4),round(rmse_arima_train,4)],
        "Gap (°C)":[round(rmse_xgb-rmse_xgb_train,4),round(rmse_h-rmse_h_train,4),round(rmse_prop-rmse_prop_train,4),
                    round(rmse_rf-rmse_rf_train,4),round(rmse_ha-rmse_ha_train,4),round(rmse_s-rmse_s_train,4),round(rmse_arima-rmse_arima_train,4)],
        "Overfitting":["Mild","Stable","Stable","Boundary","Mild","Complex Params","Misspecification"],
    }).sort_values("RMSE (°C)").reset_index(drop=True)
    comp.index = comp.index+1

    # Summary cards
    c1,c2,c3 = st.columns(3)
    with c1: metric_card("Best Accuracy", "XGBoost", f"RMSE: {rmse_xgb:.4f}°C", "🏆")
    with c2: metric_card("Best Stability", "Harmonic Reg", f"Gap: +{rmse_h-rmse_h_train:.2f}°C", "🔒")
    with c3: metric_card("Weakest Model", "ARIMA", f"RMSE: {rmse_arima:.4f}°C", "📉")

    st.markdown("<br>", unsafe_allow_html=True)
    t1,t2,t3,t4=st.tabs(["📋 Full Table","📊 Charts","🎯 Top 3 Analysis","🏆 Best Model"])

    with t1:
        st.dataframe(comp, use_container_width=True)

    with t2:
        for metric_col, title in [("RMSE (°C)","RMSE Comparison"),("MAE (°C)","MAE Comparison"),("MAPE (%)","MAPE Comparison")]:
            fig=go.Figure(go.Bar(x=comp["Model"],y=comp[metric_col],
                marker_color=["#2980b9" if t=="ML" else "#27ae60" for t in comp["Type"]],
                text=comp[metric_col].round(4),textposition="outside"))
            fig.update_layout(title={"text":title,"x":0.5},xaxis_title="Model",yaxis_title=metric_col,
                template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",height=380)
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=test.index,y=test.values,name="Actual",line=dict(color="#2c3e50",width=3)))
        fig.add_trace(go.Scatter(x=fc_s.index,y=fc_s.values,name="SARIMA",line=dict(color="#8e44ad",dash="dash",width=2)))
        fig.add_trace(go.Scatter(x=fc_h.index,y=fc_h.values,name="Harmonic Reg",line=dict(color="#2980b9",dash="dash",width=2)))
        fig.add_trace(go.Scatter(x=fc_xgb.index,y=fc_xgb.values,name="XGBoost",line=dict(color="#f39c12",dash="dash",width=2)))
        fig.update_layout(title={"text":"Top 3 Models Forecast vs Actual","x":0.5},
            xaxis_title="Date",yaxis_title="Temperature (°C)",
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",
            hovermode="x unified",height=400,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            top3_m=["XGBoost","Harmonic Reg","SARIMA"]
            top3_r=[rmse_xgb,rmse_h,rmse_s]
            top3_g=[rmse_xgb-rmse_xgb_train,rmse_h-rmse_h_train,rmse_s-rmse_s_train]
            top3_c=["#f39c12","#2980b9","#8e44ad"]
            fig2=go.Figure()
            for m,r,g,c in zip(top3_m,top3_r,top3_g,top3_c):
                fig2.add_trace(go.Scatter(x=[r],y=[g],mode="markers+text",name=m,text=[m],
                    textposition="top center",marker=dict(size=20,color=c)))
            fig2.add_vline(x=1.3,line_dash="dot",line_color="#bdc3c7",opacity=0.6)
            fig2.add_hline(y=0.65,line_dash="dot",line_color="#bdc3c7",opacity=0.6)
            fig2.update_layout(title={"text":"Accuracy vs Stability","x":0.5},
                xaxis_title="Test RMSE (°C) ← Lower is Better",
                yaxis_title="Overfitting Gap (°C) ← Lower is Better",
                template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",height=400)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            fig3=go.Figure()
            fig3.add_trace(go.Bar(name="Train RMSE",x=top3_m,
                y=[rmse_xgb_train,rmse_h_train,rmse_s_train],
                marker_color=["rgba(243,156,18,.4)","rgba(41,128,185,.4)","rgba(142,68,173,.4)"]))
            fig3.add_trace(go.Bar(name="Test RMSE",x=top3_m,y=top3_r,marker_color=top3_c))
            fig3.update_layout(title={"text":"Train vs Test RMSE","x":0.5},
                barmode="group",xaxis_title="Model",yaxis_title="RMSE (°C)",
                template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",height=400,
                legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
            st.plotly_chart(fig3, use_container_width=True)

    with t4:
        for role,model,reason,clr in [
            ("🎯 Maximum Accuracy","XGBoost",f"RMSE: {rmse_xgb:.4f}°C, MAPE: {mape_xgb:.2f}% — best raw accuracy across all 7 models","#f39c12"),
            ("🔒 Best Stability","Harmonic Regression",f"Gap: +{rmse_h-rmse_h_train:.2f}°C — smallest train-test gap; deterministic structure generalizes perfectly","#2980b9"),
            ("📐 Classical Benchmark","SARIMA","Explicit 12-month seasonal modelling — required classical approach for academic comparison","#8e44ad"),
        ]:
            st.markdown(f"""
            <div class="best-card" style="background:#fafcff;border-color:{clr};">
                <span style="color:{clr};font-size:0.85rem;font-weight:700;">{role}</span><br>
                <span style="font-size:1.2rem;font-weight:800;color:#1a2e44;">{model}</span><br>
                <span style="font-size:0.83rem;color:#5d7b8a;">{reason}</span>
            </div>
            """, unsafe_allow_html=True)
        st.warning("⚠️ **ML Evaluation Note:** RF and XGBoost use one-step-ahead evaluation (true lag values as input), giving them a structural advantage over classical models which perform genuine multi-step-ahead forecasting.")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: FUTURE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Future Forecast":
    page_header("🔮 Future Forecast", "Forecast India's monthly temperature beyond December 2017")

    col1, col2 = st.columns([1, 2])
    with col1:
        fc_model = st.selectbox("Select Model:", [
            "Harmonic Regression","SARIMA","Harmonic + ARIMA","Prophet","XGBoost","Random Forest","ARIMA"])
        horizon = st.selectbox("Horizon:", ["1 year","5 years","10 years","20 years","30 years","Custom"])
        n_years = int(horizon.split()[0]) if horizon != "Custom" else st.number_input("Years:", 1, 50, 5)
        n_steps = n_years * 12
        metric_card("Forecast Steps", f"{n_steps}", f"Jan 2018 → {pd.date_range('2018-01',periods=n_steps,freq='MS')[-1].strftime('%b %Y')}", "📅")
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="info-card" style="margin-top:0;">
            <h4>About this forecast</h4>
            <p>All models are trained on <b>Jan 1901 – Nov 1994</b> (training set).
            Forecasts start from <b>January 2018</b> — beyond the dataset's last observation.
            Prophet and ARIMA also provide 95% confidence intervals.</p>
        </div>
        """, unsafe_allow_html=True)

    if run:
        future_idx = pd.date_range("2018-01-01", periods=n_steps, freq="MS")
        conf_int = None

        with st.spinner(f"Generating {n_years}-year forecast…"):
            if fc_model == "Harmonic Regression":
                hm = fit_harmonic_model(train_harm, y_train_harm)
                t_f = np.arange(len(df), len(df)+n_steps); s2=12
                feat=pd.DataFrame({"t":t_f,
                    "cos1":np.cos(2*np.pi*1*t_f/s2),"sin1":np.sin(2*np.pi*1*t_f/s2),
                    "cos2":np.cos(2*np.pi*2*t_f/s2),"sin2":np.sin(2*np.pi*2*t_f/s2)},index=future_idx)
                fc_out=pd.Series(hm.predict(feat),index=future_idx)

            elif fc_model == "SARIMA":
                from pmdarima import auto_arima 
                # Refit on full data (1901–2017) so forecast starts from Jan 2018
                sarima_full = auto_arima(df["TEMPERATURE"], start_p=0, max_p=3, start_q=0, max_q=3,
                             d=1, seasonal=True, m=12, start_P=0, max_P=2,
                             start_Q=0, max_Q=2, D=1, stepwise=True, trace=False,
                             error_action="ignore", suppress_warnings=True)
                fc_out=pd.Series(sarima_full.predict(n_periods=n_steps),index=future_idx)

            
            elif fc_model == "Harmonic + ARIMA":
                h1m,am,_=fit_harm_arima_model(train)
                p=12; t_f=np.arange(len(df),len(df)+n_steps)  # ← len(df) មិនមែន len(train)
                h_f=pd.DataFrame({"sin":np.sin(2*np.pi*t_f/p),"cos":np.cos(2*np.pi*t_f/p)},index=future_idx)
                # Refit ARIMA residual on full residuals so it starts from 2018
                from pmdarima import auto_arima
                from sklearn.linear_model import LinearRegression
                ti_full=np.arange(len(df))
                h_full=pd.DataFrame({"sin":np.sin(2*np.pi*ti_full/p),"cos":np.cos(2*np.pi*ti_full/p)},index=df.index)
                hm_full=LinearRegression(); hm_full.fit(h_full,df["TEMPERATURE"])
                res_full=df["TEMPERATURE"]-pd.Series(hm_full.predict(h_full),index=df.index)
                am_full=auto_arima(res_full,start_p=0,max_p=5,start_q=0,max_q=5,
                       d=0,seasonal=False,stepwise=True,trace=False,
                       error_action="ignore",suppress_warnings=True)
                t_f=np.arange(len(df),len(df)+n_steps)
                h_f=pd.DataFrame({"sin":np.sin(2*np.pi*t_f/p),"cos":np.cos(2*np.pi*t_f/p)},index=future_idx)
                fc_out=pd.Series(hm_full.predict(h_f)+am_full.predict(n_periods=n_steps),index=future_idx)
                if n_years > 10:
                    st.warning("⚠️ **Harmonic+ARIMA Note:** ARIMA residual component may dampen for very long horizons. For 30-year forecasts, Harmonic Regression or SARIMA are more reliable.")

            elif fc_model == "Prophet":
                pm=fit_prophet_model(train,train_size)
                fut=pm.make_future_dataframe(periods=len(test)+n_steps,freq="MS")
                fc_full=pm.predict(fut)
                sl=fc_full[fc_full["ds"] >= "2018-01-01"].head(n_steps)
                fc_out=pd.Series(sl["yhat"].values,index=future_idx)
                conf_int=(pd.Series(sl["yhat_lower"].values,index=future_idx),
                          pd.Series(sl["yhat_upper"].values,index=future_idx))

            elif fc_model == "ARIMA":
                arima_fit,arima_order,_=fit_arima_model(train)
                fc_obj=arima_fit.get_forecast(steps=n_steps)
                fc_out=pd.Series(fc_obj.predicted_mean.values,index=future_idx)
                ci_df=fc_obj.conf_int(alpha=0.05)
                conf_int=(pd.Series(ci_df.iloc[:,0].values,index=future_idx),
                          pd.Series(ci_df.iloc[:,1].values,index=future_idx))

            elif fc_model in ["XGBoost","Random Forest"]:
                if fc_model=="XGBoost": model_=fit_xgb_model(X_train,y_train,X_test,y_test)
                else: model_=fit_rf_model(X_train,y_train)
                history=list(df["TEMPERATURE"].values[-12:])
                fc_vals=[]
                for _ in range(n_steps):
                    lag_in=np.array(history[-12:]).reshape(1,-1)
                    pred=model_.predict(lag_in)[0]
                    fc_vals.append(pred); history.append(pred)
                fc_out=pd.Series(fc_vals,index=future_idx)
                if n_years > 2:
                    st.warning("⚠️ **ML Long-horizon Note:** XGBoost/Random Forest use recursive lag prediction — forecasts converge to a constant mean beyond ~2 years. This is a known limitation of lag-based ML models for long-horizon forecasting.")

        fig=go.Figure()
        hist=df["TEMPERATURE"]["2010":]
        fig.add_trace(go.Scatter(x=hist.index,y=hist,name="Historical (2010–2017)",line=dict(color="#2980b9",width=1.5)))
        if conf_int:
            lo,hi=conf_int
            fig.add_trace(go.Scatter(
                x=list(fc_out.index)+list(fc_out.index[::-1]),
                y=list(hi)+list(lo[::-1]),
                fill="toself",fillcolor="rgba(231,76,60,0.1)",
                line=dict(color="rgba(255,255,255,0)"),name="95% CI"))
        fig.add_trace(go.Scatter(x=fc_out.index,y=fc_out,name=f"{fc_model} Forecast",
            line=dict(color="#e74c3c",width=2,dash="dash")))
        
        fig.add_vline(x=pd.Timestamp("2018-01-01").timestamp()*1000,
            line_dash="dot",line_color="#27ae60",
            annotation_text="Forecast Start",annotation_position="top right")



        fig.update_layout(
            title={"text":f"{fc_model} — {n_years}-Year Forecast (Jan 2018 onwards)","x":0.5},
            xaxis_title="Date",yaxis_title="Temperature (°C)",
            template="plotly_white",paper_bgcolor="white",plot_bgcolor="#fafcff",
            hovermode="x unified",height=480,
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        st.plotly_chart(fig, use_container_width=True)

        fc_df=pd.DataFrame({"Date":fc_out.index,"Year":fc_out.index.year,
            "Month":fc_out.index.strftime("%b"),"Forecast (°C)":fc_out.values.round(4)})
        if conf_int:
            fc_df["Lower 95%"]=conf_int[0].values.round(4)
            fc_df["Upper 95%"]=conf_int[1].values.round(4)

        c1, c2 = st.columns([1.5, 1])
        with c1:
            section_title("📋 Forecast Table")
            st.dataframe(fc_df, use_container_width=True, hide_index=True)
            st.download_button("📥 Download Forecast CSV",
                fc_df.to_csv(index=False).encode(),
                f"forecast_{fc_model.lower().replace(' ','_')}_{n_years}yr.csv","text/csv")
        with c2:
            section_title("📊 Annual Summary")
            annual=fc_df.groupby("Year")["Forecast (°C)"].agg(["mean","min","max"]).round(2)
            annual.columns=["Mean (°C)","Min (°C)","Max (°C)"]
            st.dataframe(annual, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    page_header("ℹ️ About", "Project summary, methodology, and author information")

    col1, col2 = st.columns([1.2, 0.8])
    with col1:
        section_title("📌 Project Summary")
        st.markdown("""
        Comprehensive time series forecasting on India's monthly average temperature (1901–2017).
        Seven models from classical statistical and machine learning paradigms evaluated side-by-side
        with explicit attention to seasonal misspecification, multi-step forecast degradation, and
        train–test generalization gaps.
        """)
        section_title("🧪 Methodology")
        steps = [
            ("Data Preparation", "Wide CSV reshaped to monthly time series with DatetimeIndex"),
            ("EDA", "Visualization, seasonal decomposition, rolling statistics, Kruskal-Wallis test"),
            ("Stationarity", "ADF test confirms d=1; ACF/PACF guides ARIMA order selection"),
            ("Train/Test Split", "Chronological 80/20 — Jan 1901–Nov 1994 / Dec 1994–Dec 2017"),
            ("Model Fitting", "7 models fitted on training data only"),
            ("Evaluation", "MAE, RMSE, MAPE on 23-year held-out test set"),
            ("Overfitting Analysis", "Train vs test RMSE gap for all models"),
            ("Diagnostics", "4-panel residual diagnostics for ARIMA, Harmonic+ARIMA, SARIMA"),
        ]
        for i, (title, desc) in enumerate(steps, 1):
            st.markdown(f"""
            <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:8px;'>
                <span style='background:#2980b9;color:white;border-radius:50%;width:22px;height:22px;
                display:flex;align-items:center;justify-content:center;font-size:0.72rem;
                font-weight:700;flex-shrink:0;'>{i}</span>
                <span style='font-size:0.85rem;color:#2c3e50;'><b>{title}</b> — {desc}</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        section_title("👤 Author")
        info = [("Name","SEANG Vutha"),("ID","M080405"),("Program","M-DAS"),
                ("Course","Time Series Analysis and Forecasting"),
                ("Instructor","Dr. NHIM Malai"),("Institution","ITC — Graduate School"),("Year","2025–2026")]
        for k, v in info:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;padding:6px 0;
            border-bottom:1px solid #f0f0f0;font-size:0.83rem;'>
                <span style='color:#7f8c9a;font-weight:600;'>{k}</span>
                <span style='color:#1a2e44;font-weight:500;'>{v}</span>
            </div>
            """, unsafe_allow_html=True)

        section_title("📚 Libraries")
        libs = [("streamlit","Dashboard"),("pandas","Data manipulation"),("numpy","Numerics"),
                ("plotly","Charts"),("statsmodels","ARIMA/ADF/decomposition"),
                ("pmdarima","auto_arima"),("scikit-learn","Linear Regression / RF"),
                ("xgboost","XGBoost"),("prophet","Prophet"),("scipy","Statistics"),
                ("kagglehub","Dataset download")]
        st.dataframe(pd.DataFrame(libs, columns=["Library","Purpose"]),
            use_container_width=True, hide_index=True)