"""
app.py
AeroGraph — Autonomous Air Quality Orchestrator
Streaming Incremental Machine Learning (River) + LangGraph Multi-Persona Reasoning.
"""

from datetime import datetime
import json
import os
from typing import Any, Dict, List
import pandas as pd
import plotly.graph_objects as go
import pytz
import streamlit as st

from streamlit_autorefresh import st_autorefresh

from graph.workflow import run_aerograph_orchestration
from ml.online_model import OnlineMLResult, train_and_forecast_online
from tools.data_engine import (
    PRESET_CITIES,
    fetch_ground_truth_validation,
    fetch_live_data,
    fetch_monthly_history,
    get_aqi_category_info,
)

# Streamlit Page Configuration: expanded layout, top controls
st.set_page_config(
    page_title="AeroGraph | Autonomous AQI Orchestrator",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# Helper: Compact single-line HTML renderer to guarantee zero Markdown code-block interpretation
def render_html(html_str: str):
    """Renders HTML safely in Streamlit as a single contiguous string stream,

    completely preventing any Markdown 4-space code block interpretation.
    """
    compact = " ".join(
        [line.strip() for line in html_str.splitlines() if line.strip()]
    )
    st.markdown(compact, unsafe_allow_html=True)


# Custom CSS: Modern Warm Off-White Aesthetic & Full-Width Top Navigation
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* Force Light Color Scheme across all browsers & OS dark-mode preferences */
:root, [data-theme="dark"], [data-theme="light"], .stApp, html, body {
    color-scheme: light !important;
}

@media (prefers-color-scheme: dark) {
    :root, html, body, .stApp, [data-testid="stAppViewContainer"] {
        color-scheme: light !important;
        background-color: #F8F9FA !important;
        color: #0F172A !important;
    }
}

/* Global Typography & Palette */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: #1E293B !important;
    background-color: #F8F9FA !important;
}

/* Ensure no text defaults to white on off-white background */
p, label, div[data-testid="stMarkdownContainer"] p {
    color: #1E293B !important;
}

span:not([style*="color"]):not(.ag-badge) {
    color: #1E293B !important;
}

h1, h2, h3, h4, h5, h6,
div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4 {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Hide Streamlit default sidebar collapse chevron if any */
[data-testid="collapsedControl"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background-color: #F8F9FA !important;
}

/* Native Bordered Containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    padding: 16px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: #FFFFFF !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] span:not([style*="color"]):not(.ag-badge) {
    color: #1E293B !important;
}

/* Native Metric Cards & Labels */
[data-testid="stMetric"] {
    background: transparent !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #64748B !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color: #0F172A !important;
    font-weight: 800 !important;
    font-size: 1.85rem !important;
    line-height: 1.2 !important;
}

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span,
.stCaption {
    color: #64748B !important;
}

/* Widgets & Form Controls */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span {
    color: #0F172A !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
}

.stRadio div[role="radiogroup"] label {
    color: #1E293B !important;
}
.stRadio div[role="radiogroup"] p,
.stRadio div[role="radiogroup"] span {
    color: #1E293B !important;
    font-weight: 500 !important;
}

[data-baseweb="select"] {
    background-color: #FFFFFF !important;
}
[data-baseweb="select"] div,
[data-baseweb="select"] span,
[data-baseweb="select"] p {
    color: #0F172A !important;
    background-color: transparent !important;
}
div[data-baseweb="popover"],
ul[data-baseweb="menu"],
li[data-baseweb="menu-item"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
}
li[data-baseweb="menu-item"]:hover {
    background-color: #F1F5F9 !important;
    color: #0F172A !important;
}

[data-testid="stTextInput"] input {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 8px !important;
}
[data-testid="stTextInput"] input::placeholder {
    color: #94A3B8 !important;
}

.stToggle label span,
.stCheckbox label span {
    color: #0F172A !important;
    font-weight: 600 !important;
}

.stButton > button {
    background-color: #0F172A !important;
    color: #FFFFFF !important;
    border: 1px solid #0F172A !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background-color: #1E293B !important;
    color: #FFFFFF !important;
    border-color: #1E293B !important;
}
.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

/* Custom Metric Cards */
.ag-metric-card {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    padding: 16px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
    text-align: left !important;
}

.ag-metric-label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    color: #64748B !important;
    margin-bottom: 4px !important;
}

.ag-metric-value {
    font-size: 1.65rem !important;
    font-weight: 800 !important;
    color: #0F172A !important;
    line-height: 1.2 !important;
}

.ag-metric-sub {
    font-size: 0.8rem !important;
    color: #64748B !important;
    margin-top: 4px !important;
}
.ag-metric-sub b {
    color: #0F172A !important;
}

/* Badges */
.ag-badge {
    display: inline-flex !important;
    align-items: center !important;
    padding: 4px 12px !important;
    border-radius: 9999px !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
}

/* Banner Alert Box */
.ag-alert-strip {
    border-radius: 12px !important;
    padding: 18px 24px !important;
    margin-bottom: 20px !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 10px !important;
    border-left: 6px solid !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}

.ag-alert-normal {
    background: #ECFDF5 !important;
    border-color: #10B981 !important;
}
.ag-alert-normal .ag-alert-title {
    color: #065F46 !important;
}

.ag-alert-advisory {
    background: #FFFBEB !important;
    border-color: #F59E0B !important;
}
.ag-alert-advisory .ag-alert-title {
    color: #92400E !important;
}

.ag-alert-critical {
    background: #FEF2F2 !important;
    border-color: #EF4444 !important;
}
.ag-alert-critical .ag-alert-title {
    color: #991B1B !important;
}

.ag-alert-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
}
.ag-alert-title span {
    color: inherit !important;
}

.ag-alert-desc {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    color: #1E293B !important;
    background: rgba(255, 255, 255, 0.92) !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    border: 1px solid rgba(0, 0, 0, 0.06) !important;
}
.ag-alert-desc b, .ag-alert-desc strong {
    color: #0F172A !important;
}

/* Modern Persona Tabs: Full Width at Top, zero side arrows */
.stTabs {
    width: 100% !important;
}

.stTabs [data-baseweb="tab-list"] {
    display: flex !important;
    width: 100% !important;
    gap: 10px !important;
    background: #FFFFFF !important;
    padding: 8px !important;
    border-radius: 12px !important;
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03) !important;
    overflow-x: auto !important;
    scrollbar-width: none !important;
}

.stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {
    display: none !important;
}

.stTabs button[aria-label="Previous"],
.stTabs button[aria-label="Next"],
.stTabs [data-testid="stTabsScrollButton"] {
    display: none !important;
}

.stTabs [data-baseweb="tab"] {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
    border: 1px solid #E2E8F0 !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    white-space: nowrap !important;
}

/* Unselected Tabs */
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) {
    background: #F8FAFC !important;
    border-color: #E2E8F0 !important;
}
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) p,
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) span,
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) div {
    color: #334155 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

.stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover {
    background: #EDF2F7 !important;
    border-color: #CBD5E1 !important;
}
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover p,
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]):hover span {
    color: #0F172A !important;
}

/* Active Selected Tab */
.stTabs [aria-selected="true"] {
    background: #0F172A !important;
    border-color: #0F172A !important;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15) !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* Directive rows inside tabs */
.ag-directive-row {
    margin-bottom: 14px !important;
}

.ag-directive-label {
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    color: #475569 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    margin-bottom: 4px !important;
}

.ag-directive-value {
    font-size: 0.95rem !important;
    color: #0F172A !important;
    font-weight: 500 !important;
    line-height: 1.55 !important;
    background: #F8FAFC !important;
    padding: 12px 16px !important;
    border-radius: 8px !important;
    border: 1px solid #E2E8F0 !important;
}

.ag-directive-value b, .ag-directive-value strong {
    color: #0F172A !important;
    font-weight: 700 !important;
}

/* Alert notifications (st.success, st.error, etc.) */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] span,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] strong,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] b {
    color: #0F172A !important;
}

/* Agent Trace Expander */
[data-testid="stExpander"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary * {
    color: #0F172A !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary svg {
    fill: #0F172A !important;
}
[data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
}
[data-testid="stCodeBlock"] {
    background: #0F172A !important;
    border-radius: 8px !important;
    border: 1px solid #1E293B !important;
}
[data-testid="stCodeBlock"] pre,
[data-testid="stCodeBlock"] code,
[data-testid="stCodeBlock"] span {
    color: #38BDF8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

hr {
    border-color: #E2E8F0 !important;
    margin: 14px 0 !important;
}
</style>
"""

render_html(CUSTOM_CSS)


# Helper: Format Plotly Timeseries Chart with continuous chronology
def build_aqi_forecast_chart(
    df: pd.DataFrame,
    ml_result: OnlineMLResult,
    forecast_points: List[Any],
    current_aqi: float,
) -> go.Figure:
    """Builds an elegant Plotly line chart comparing the last 24h actual readings,

    the online River fitted track, and the 3-hour forward forecast in an off-white layout.
    """
    df_recent = df.tail(24).copy()

    # Format timestamps chronologically with date and time: e.g. "10 Oct 14:00"
    actual_dts = [pd.to_datetime(t) for t in df_recent["time"]]
    actual_labels = [dt.strftime("%d %b %H:%M") for dt in actual_dts]
    actual_aqis = df_recent["aqi"].tolist()

    # Connect from the latest actual point to the forward forecast points
    forecast_dts = [pd.to_datetime(f.timestamp) for f in forecast_points]
    forecast_labels = [
        f"{dt.strftime('%d %b %H:%M')} ({f.step_label})"
        for dt, f in zip(forecast_dts, forecast_points)
    ]
    forecast_aqis = [f.predicted_aqi for f in forecast_points]

    conn_labels = [actual_labels[-1]] + forecast_labels
    conn_aqis = [actual_aqis[-1]] + forecast_aqis

    fig = go.Figure()

    # 1. Past 24h Actual CPCB AQI
    fig.add_trace(
        go.Scatter(
            x=actual_labels,
            y=actual_aqis,
            mode="lines+markers",
            name="Actual CPCB AQI (Past 24h)",
            line=dict(color="#2563EB", width=2.8),
            marker=dict(size=5, color="#1D4ED8"),
            hovertemplate=(
                "%{x}<br>Actual CPCB AQI: <b>%{y:.1f}</b><extra></extra>"
            ),
        )
    )

    # 2. River Online Model Fitted Track (over past 24h)
    if ml_result.eval_history:
        eval_df = pd.DataFrame(ml_result.eval_history).tail(24)
        if len(eval_df) > 0:
            eval_labels = [
                pd.to_datetime(t).strftime("%d %b %H:%M")
                for t in eval_df["time"]
            ]
            fig.add_trace(
                go.Scatter(
                    x=eval_labels,
                    y=eval_df["model_pred"].tolist(),
                    mode="lines",
                    name="River Model Fitted Track",
                    line=dict(color="#10B981", width=2, dash="dot"),
                    hovertemplate=(
                        "%{x}<br>River Model Track:"
                        " <b>%{y:.1f}</b><extra></extra>"
                    ),
                )
            )

    # 3. 3-Hour Forward Prediction Horizon (t+1, t+2, t+3)
    fig.add_trace(
        go.Scatter(
            x=conn_labels,
            y=conn_aqis,
            mode="lines+markers",
            name="River 3h Forward Forecast",
            line=dict(color="#EF4444", width=3, dash="dash"),
            marker=dict(size=8, color="#DC2626", symbol="diamond"),
            hovertemplate=(
                "%{x}<br>Forecasted AQI: <b>%{y:.1f}</b><extra></extra>"
            ),
        )
    )

    # Threshold horizontal reference lines
    fig.add_hline(
        y=300,
        line_dash="dot",
        line_color="#94A3B8",
        annotation_text="Very Poor (300)",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#64748B",
    )
    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="#CBD5E1",
        annotation_text="Satisfactory (100)",
        annotation_position="bottom right",
        annotation_font_size=10,
        annotation_font_color="#94A3B8",
    )

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=20, b=40),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=10, color="#64748B"),
            title=None,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=11, color="#64748B"),
            title=dict(text="CPCB AQI", font=dict(size=12, color="#475569")),
        ),
        hovermode="x unified",
    )

    return fig


# Helper: Format 30-Day Historical Monthly Trend Chart with CPCB Severity Bands
def build_monthly_trend_chart(
    df_monthly: pd.DataFrame, city_name: str
) -> go.Figure:
    """Builds an interactive Plotly chart displaying the 30-day historical AQI trajectory

    with color-coded CPCB severity background bands.
    """
    fig = go.Figure()

    bands = [
        (0, 50, "rgba(16, 185, 129, 0.08)", "Good"),
        (50, 100, "rgba(52, 211, 153, 0.08)", "Satisfactory"),
        (100, 200, "rgba(245, 158, 11, 0.08)", "Moderate"),
        (200, 300, "rgba(249, 115, 22, 0.08)", "Poor"),
        (300, 400, "rgba(239, 68, 68, 0.08)", "Very Poor"),
        (400, 500, "rgba(127, 29, 29, 0.08)", "Severe"),
    ]
    for y0, y1, color, label in bands:
        fig.add_hrect(
            y0=y0,
            y1=y1,
            fillcolor=color,
            layer="below",
            line_width=0,
        )

    # Format dates
    dates = [
        pd.to_datetime(t).strftime("%d %b %H:%M") for t in df_monthly["time"]
    ]
    aqi_values = [round(float(v), 1) for v in df_monthly["aqi"]]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=aqi_values,
            mode="lines",
            name=f"30-Day CPCB AQI ({city_name})",
            line=dict(color="#0284C7", width=2.4),
            hovertemplate="%{x}<br>CPCB AQI: <b>%{y:.1f}</b><extra></extra>",
        )
    )

    fig.add_hline(
        y=100,
        line_dash="dot",
        line_color="#10B981",
        annotation_text="Satisfactory (100)",
        annotation_position="top right",
        annotation_font_size=9,
        annotation_font_color="#059669",
    )
    fig.add_hline(
        y=300,
        line_dash="dot",
        line_color="#EF4444",
        annotation_text="Very Poor (300)",
        annotation_position="top right",
        annotation_font_size=9,
        annotation_font_color="#DC2626",
    )

    max_val = max(aqi_values) if aqi_values else 200
    max_y = max(350, max_val + 30)

    fig.update_layout(
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#0F172A"),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(l=40, r=20, t=20, b=40),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#475569"),
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=10, color="#64748B"),
            nticks=10,
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=11, color="#64748B"),
            title=dict(text="CPCB AQI", font=dict(size=12, color="#475569")),
            range=[0, max_y],
        ),
        hovermode="x unified",
    )

    return fig


# Auto-refresh periodically every 5 minutes (300,000 ms)
st_autorefresh(interval=5 * 60 * 1000, key="aerograph_5min_refresh")

bedrock_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Application Header & Advisory Notice
render_html(f"""
<div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 8px;">
    <div>
        <h1 style="font-size: 1.85rem; font-weight: 800; color: #0F172A; margin: 0; letter-spacing: -0.02em;">
            AeroGraph <span style="font-weight: 400; color: #64748B; font-size: 1.3rem;">| Autonomous Air Quality Orchestrator</span>
        </h1>
        <p style="color: #64748B; font-size: 0.92rem; margin: 4px 0 0 0;">
            Amazon Bedrock (Nova Pro) &middot; AWS Strands Agents SDK &middot; Streaming ML (River) &middot; Ground-Truth Cross Validation
        </p>
    </div>
</div>
<div style="background-color: #FEF3C7; border: 1px solid #FDE68A; border-radius: 8px; padding: 10px 16px; margin-bottom: 16px; font-size: 0.82rem; color: #92400E; display: flex; align-items: center; gap: 8px;">
    <span>⚠️</span>
    <span><b>Official Advisory Notice:</b> Action directives are rule-bounded decision-support recommendations generated using Indian CPCB sub-index standards and do not substitute certified medical, institutional, or regulatory authority directives.</span>
</div>
""")


# REDESIGNED TOP CONTROL BAR: AWS Bedrock & Strands Agents Integration
with st.container(border=True):
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4, col_ctrl5 = st.columns(
        [1.8, 2.1, 2.1, 2.0, 1.0]
    )

    with col_ctrl1:
        city_selected = st.selectbox(
            "🏙️ Target City",
            options=list(PRESET_CITIES.keys()),
            index=0,
            help="10 Major Indian Metropolitans for atmospheric streaming and ground telemetry pre-training",
        )

    with col_ctrl2:
        st.write('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        st.markdown("**☁️ AI Backend**")
        st.caption("Amazon Bedrock (Nova Pro Serverless)")

    with col_ctrl3:
        st.write('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        st.markdown("**🤖 Agent Framework**")
        st.caption("AWS Strands Agents SDK")

    with col_ctrl4:
        st.write('<div style="height: 4px;"></div>', unsafe_allow_html=True)
        st.markdown(f"**🌐 AWS Region & Sync**")
        st.caption(f"Region: {bedrock_region} &middot; Refresh: 5m")

    with col_ctrl5:
        st.write(
            '<div style="height: 28px;"></div>', unsafe_allow_html=True
        )  # vertical alignment
        btn_refresh = st.button("🔄 Refresh", width="stretch")


# Telemetry Data Loading & Computation
with st.spinner(
    f"Streaming live atmospheric telemetry & warm-starting River model for"
    f" {city_selected}..."
):
    df_raw = fetch_live_data(city_selected)
    active_city_name = city_selected

    # Execute River Online ML with multi-year ground pre-training
    ml_result: OnlineMLResult = train_and_forecast_online(
        df_raw, city=city_selected
    )

    latest_row = df_raw.iloc[-1]
    current_aqi = float(ml_result.current_aqi)
    dominant_pollutant = str(latest_row["dominant_pollutant"])
    pm25_val = float(latest_row["pm2_5"])
    pm10_val = float(latest_row["pm10"])
    no2_val = float(latest_row["nitrogen_dioxide"])
    so2_val = float(latest_row["sulphur_dioxide"])
    wind_spd = float(latest_row["wind_speed_10m"])
    wind_deg = float(latest_row["wind_direction_10m"])
    hum_val = float(latest_row["relative_humidity_2m"])

    # Independent ground-truth validation check (WAQI / CPCB station feed)
    coords = PRESET_CITIES[city_selected]
    val_result = fetch_ground_truth_validation(
        coords["lat"], coords["lon"], city_selected, current_aqi
    )

    # Prepare forecast dictionary format for state graph
    forecast_dicts = [
        {
            "timestamp": pt.timestamp,
            "step_label": pt.step_label,
            "predicted_aqi": pt.predicted_aqi,
            "delta": pt.delta_from_current,
        }
        for pt in ml_result.forecast_trajectory
    ]

    # Execute LangGraph Reasoning Workflow
    agent_state = run_aerograph_orchestration(
        city=active_city_name,
        current_aqi=current_aqi,
        current_pm25=pm25_val,
        current_pm10=pm10_val,
        current_no2=no2_val,
        current_so2=so2_val,
        wind_speed=wind_spd,
        wind_direction=wind_deg,
        humidity=hum_val,
        dominant_pollutant=dominant_pollutant,
        forecast_trajectory=forecast_dicts,
        is_breach=ml_result.is_breach,
        breach_deltas=ml_result.breach_deltas,
        model_mae=ml_result.model_mae,
        persistence_mae=ml_result.persistence_mae,
    )

playbook = agent_state.get("playbook", {})
used_cache = agent_state.get("used_cache", False)
diag_prof = agent_state.get("diagnostic_profile", {})
alert_level = playbook.get("alert_level", "Normal Airflow")
root_cause = playbook.get(
    "root_cause_summary", "Atmospheric background dispersion within norms."
)


# Top Section: Cross-Source Ground-Truth Verification & Dynamic Alert Strip
render_html(f"""
<div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 12px 18px; margin-bottom: 14px; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);">
    <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-size: 1.25rem;">🛰️</span>
        <div>
            <div style="font-size: 0.9rem; font-weight: 700; color: #0F172A;">
                Independent Cross-Source AQI Ground Verification
            </div>
            <div style="font-size: 0.82rem; color: #475569; margin-top: 2px;">
                CPCB Calculated Engine: <b style="color: #0F172A;">{current_aqi:.0f}</b> &nbsp;|&nbsp; Ground Reference Sensor: <b style="color: #0F172A;">{val_result['ground_aqi']:.0f}</b> &nbsp;|&nbsp; Variance Delta: <b style="color: {val_result['badge_color']};">{val_result['delta_pct']}%</b>
            </div>
        </div>
    </div>
    <div>
        <span class="ag-badge" style="background: {val_result['badge_color']}15; color: {val_result['badge_color']}; border: 1px solid {val_result['badge_color']}44; font-size: 0.82rem;">
            ✓ {val_result['status']} &middot; {val_result['source']}
        </span>
    </div>
</div>
""")

# Helper to rank CPCB severity band
def get_cpcb_band_rank(aqi_val: float) -> int:
    if aqi_val <= 50:
        return 0
    elif aqi_val <= 100:
        return 1
    elif aqi_val <= 200:
        return 2
    elif aqi_val <= 300:
        return 3
    elif aqi_val <= 400:
        return 4
    else:
        return 5

# Compute 3-Hour Predictive Horizon Metrics & Delta Signatures
pred_t1 = forecast_dicts[0]["predicted_aqi"] if forecast_dicts else current_aqi
pred_t2 = forecast_dicts[1]["predicted_aqi"] if len(forecast_dicts) > 1 else current_aqi
pred_t3 = forecast_dicts[2]["predicted_aqi"] if len(forecast_dicts) > 2 else current_aqi
d_t1 = pred_t1 - current_aqi
d_t2 = pred_t2 - current_aqi
d_t3 = pred_t3 - current_aqi

curr_cat_info = get_aqi_category_info(current_aqi)
t1_cat_info = get_aqi_category_info(pred_t1)
t2_cat_info = get_aqi_category_info(pred_t2)
t3_cat_info = get_aqi_category_info(pred_t3)

curr_rank = get_cpcb_band_rank(current_aqi)
max_pred_aqi = max(pred_t1, pred_t2, pred_t3)
max_pred_rank = max(get_cpcb_band_rank(pred_t1), get_cpcb_band_rank(pred_t2), get_cpcb_band_rank(pred_t3))
max_surge_delta = max_pred_aqi - current_aqi

# Hero Condition: Predicted AQI rises by >= 15 points OR crosses into higher CPCB band OR consecutive breach
is_surge_predicted = (max_surge_delta >= 15.0) or (max_pred_rank > curr_rank) or ml_result.is_breach

if is_surge_predicted:
    # Early-Warning Banner (Amber/Red)
    if max_pred_aqi > 250 or ml_result.is_breach:
        strip_class = "ag-alert-critical"
        pill_bg = "#FEE2E2"
        pill_color = "#991B1B"
    else:
        strip_class = "ag-alert-advisory"
        pill_bg = "#FEF3C7"
        pill_color = "#92400E"

    if pred_t2 >= pred_t1 and pred_t2 >= pred_t3:
        target_pred_val = pred_t2
        window_str = "next 2 hours"
    elif pred_t1 >= pred_t2 and pred_t1 >= pred_t3:
        target_pred_val = pred_t1
        window_str = "next hour"
    else:
        target_pred_val = pred_t3
        window_str = "next 3 hours"

    strip_title = f"🚨 AI Inflow Warning: AQI projected to climb from {current_aqi:.0f} to {target_pred_val:.0f} within the {window_str}."
    trend_pill_html = f'<span style="background: {pill_bg}; color: {pill_color}; border: 1px solid {pill_color}44; font-size: 0.8rem; font-weight: 700; padding: 5px 14px; border-radius: 9999px; white-space: nowrap;">+{max_surge_delta:.0f} AQI &middot; Surge Expected</span>'
else:
    # Reassuring Green predictive banner
    strip_class = "ag-alert-normal"
    pill_bg = "#D1FAE5"
    pill_color = "#065F46"

    strip_title = "🟢 Stable Forecast: Airflow dispersion projected to maintain or lower AQI over the next 3 hours."
    if d_t3 < -5:
        trend_pill_html = f'<span style="background: {pill_bg}; color: {pill_color}; border: 1px solid {pill_color}44; font-size: 0.8rem; font-weight: 700; padding: 5px 14px; border-radius: 9999px; white-space: nowrap;">{d_t3:.0f} AQI &middot; Airflow Dispersing</span>'
    else:
        trend_pill_html = f'<span style="background: {pill_bg}; color: {pill_color}; border: 1px solid {pill_color}44; font-size: 0.8rem; font-weight: 700; padding: 5px 14px; border-radius: 9999px; white-space: nowrap;">±{abs(max_surge_delta):.0f} AQI &middot; Stable Dispersion</span>'

cache_notice_html = (
    '<span style="margin-left: auto; background: #EDE9FE; color: #5B21B6;'
    " font-size: 0.72rem; padding: 4px 10px; border-radius: 9999px;"
    ' font-weight: 600; border: 1px solid #DDD6FE;">⚡ Contextual Generator (Cached/Offline)</span>'
    if used_cache
    else '<span style="margin-left: auto; background: #E0F2FE; color: #0369A1;'
    " font-size: 0.72rem; padding: 4px 10px; border-radius: 9999px;"
    ' font-weight: 600; border: 1px solid #BAE6FD;">⚡ Amazon Bedrock (Nova Pro Serverless)</span>'
)

render_html(f"""
<div class="ag-alert-strip {strip_class}" style="padding: 16px 20px !important;">
    <div class="ag-alert-title" style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <span style="display: flex; align-items: center; gap: 8px;">{strip_title}</span>
        {cache_notice_html}
    </div>
    
    <!-- Hero Feature: Dedicated Predictive Horizon Strip -->
    <div style="background: #FFFFFF; border: 1px solid rgba(0,0,0,0.08); border-radius: 10px; padding: 12px 16px; margin: 6px 0; display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.02);">
        <div style="font-size: 0.76rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.05em;">
            🔮 Predictive Horizon Strip
        </div>
        <div style="display: flex; align-items: center; gap: 8px; flex-wrap: wrap;">
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 12px; text-align: center; min-width: 90px;">
                <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase;">Current</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">{current_aqi:.0f} <span style="font-size: 0.72rem; color: {curr_cat_info['color']}; font-weight: 700;">({curr_cat_info['category']})</span></div>
            </div>
            <span style="color: #94A3B8; font-weight: 800; font-size: 1.1rem;">➔</span>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 12px; text-align: center; min-width: 90px;">
                <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase;">+1 Hour</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">{pred_t1:.0f} <span style="font-size: 0.72rem; color: {t1_cat_info['color']}; font-weight: 700;">({d_t1:+.0f})</span></div>
            </div>
            <span style="color: #94A3B8; font-weight: 800; font-size: 1.1rem;">➔</span>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 12px; text-align: center; min-width: 90px;">
                <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase;">+2 Hours</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">{pred_t2:.0f} <span style="font-size: 0.72rem; color: {t2_cat_info['color']}; font-weight: 700;">({d_t2:+.0f})</span></div>
            </div>
            <span style="color: #94A3B8; font-weight: 800; font-size: 1.1rem;">➔</span>
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 6px 12px; text-align: center; min-width: 90px;">
                <div style="font-size: 0.68rem; color: #64748B; font-weight: 600; text-transform: uppercase;">+3 Hours</div>
                <div style="font-size: 1.05rem; font-weight: 800; color: #0F172A;">{pred_t3:.0f} <span style="font-size: 0.72rem; color: {t3_cat_info['color']}; font-weight: 700;">({d_t3:+.0f})</span></div>
            </div>
        </div>
        <div>
            {trend_pill_html}
        </div>
    </div>
    
    <div class="ag-alert-desc">
        <b>Scientific Root-Cause Diagnosis:</b> {root_cause}
    </div>
</div>
""")


# Middle Section: Metric Cards
col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)

cat_info = get_aqi_category_info(current_aqi)

with col_m1:
    render_html(f"""
    <div class="ag-metric-card" style="border-left: 4px solid {cat_info['color']};">
        <div class="ag-metric-label">CPCB AQI ({active_city_name.split()[0]})</div>
        <div class="ag-metric-value" style="color: {cat_info['color']};">{current_aqi:.0f}</div>
        <div class="ag-metric-sub">
            <span class="ag-badge" style="background: {cat_info['bg']}; color: {cat_info['text']}; border: 1px solid {cat_info['border']};">
                {cat_info['category']}
            </span>
        </div>
    </div>
    """)

with col_m2:
    render_html(f"""
    <div class="ag-metric-card">
        <div class="ag-metric-label">Dominant Driver</div>
        <div class="ag-metric-value">{dominant_pollutant}</div>
        <div class="ag-metric-sub">Sub-Index: <b>{current_aqi:.0f}</b></div>
    </div>
    """)

with col_m3:
    ratio_val = pm25_val / max(1.0, pm10_val)
    render_html(f"""
    <div class="ag-metric-card">
        <div class="ag-metric-label">PM2.5 / PM10</div>
        <div class="ag-metric-value">{pm25_val:.0f} <span style="font-size: 0.9rem; color: #64748B;">/ {pm10_val:.0f}</span></div>
        <div class="ag-metric-sub">Ratio: <b>{ratio_val:.2f}</b> µg/m³</div>
    </div>
    """)

with col_m4:
    render_html(f"""
    <div class="ag-metric-card">
        <div class="ag-metric-label">NO2 & SO2</div>
        <div class="ag-metric-value">{no2_val:.0f} <span style="font-size: 0.9rem; color: #64748B;">/ {so2_val:.0f}</span></div>
        <div class="ag-metric-sub">Precursor µg/m³</div>
    </div>
    """)

with col_m5:
    cardinal_wind = diag_prof.get("upwind_bearing_label", f"{int(wind_deg)}°")
    render_html(f"""
    <div class="ag-metric-card">
        <div class="ag-metric-label">Surface Wind</div>
        <div class="ag-metric-value">{wind_spd:.1f} <span style="font-size: 0.9rem; color: #64748B;">km/h</span></div>
        <div class="ag-metric-sub">Origin: <b>{cardinal_wind}</b></div>
    </div>
    """)

with col_m6:
    temp_val = latest_row.get("temperature_2m", 24.0)
    render_html(f"""
    <div class="ag-metric-card">
        <div class="ag-metric-label">Humidity / Temp</div>
        <div class="ag-metric-value">{hum_val:.0f}%</div>
        <div class="ag-metric-sub">Ambient: <b>{temp_val:.1f}°C</b></div>
    </div>
    """)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# Middle Section: Plotly Visualization & Native Streamlit ML Scorecard
col_chart, col_scorecard = st.columns([2.3, 1.1])

with col_chart:
    with st.container(border=True):
        st.markdown(
            "**📈 24-Hour Telemetry Track & 3-Hour Predictive Horizon**"
        )
        chart_fig = build_aqi_forecast_chart(
            df_raw, ml_result, ml_result.forecast_trajectory, current_aqi
        )
        st.plotly_chart(chart_fig, width="stretch")

with col_scorecard:
    with st.container(border=True):
        st.markdown("### 🤖 Online ML Scorecard")
        st.caption("River SGD Pipeline vs Persistence Baseline")

        render_html(f"""
        <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 8px 12px; margin-bottom: 12px; font-size: 0.76rem; color: #334155;">
            🏷️ <b>Pre-Trained:</b> {ml_result.pretrain_samples:,} ground records from {ml_result.pretrain_source} + Real-Time Streaming SGD Adaptation.
        </div>
        """)

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric(
                label="River Model MAE",
                value=f"{ml_result.model_mae:.2f}",
                delta=f"{ml_result.mae_improvement_pct:+.1f}% vs Baseline",
            )
        with col_s2:
            st.metric(
                label="Persistence Baseline",
                value=f"{ml_result.persistence_mae:.2f}",
                delta="Reference",
                delta_color="off",
            )

        st.markdown("---")
        if ml_result.is_breach:
            st.error(
                f"🚨 **ANOMALY BREACH TRIGGERED**\n\nForecast deltas: t+1"
                f" ({ml_result.breach_deltas[0]:+.1f}) & t+2"
                f" ({ml_result.breach_deltas[1]:+.1f}) exceed +25 threshold."
            )
        else:
            st.success(
                f"✅ **Normal Dispersion Envelope**\n\nForecast deltas: t+1"
                f" ({ml_result.breach_deltas[0]:+.1f}) & t+2"
                f" ({ml_result.breach_deltas[1]:+.1f}) within safe limits."
            )

# New Dedicated Section: 30-Day Monthly Historical AQI Trajectory Chart
st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
with st.container(border=True):
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(
            f"**📅 30-Day Monthly Historical AQI Trajectory ({city_selected})**"
        )
        st.caption(
            "Multi-day atmospheric trajectory with color-coded official Indian"
            " CPCB severity bands"
        )
    with col_hdr2:
        render_html(
            '<div style="text-align: right;"><span class="ag-badge"'
            ' style="background: #EFF6FF; color: #1E40AF; border: 1px'
            ' solid #BFDBFE;">30-Day Regional Trend</span></div>'
        )

    df_monthly = fetch_monthly_history(city_selected)
    monthly_chart = build_monthly_trend_chart(df_monthly, city_selected)
    st.plotly_chart(monthly_chart, width="stretch")


# Bottom Section: The Hero Feature — Four Persona Action Cards
st.markdown(
    """
<div style="margin-top: 18px; margin-bottom: 12px;">
    <h2 style="font-size: 1.35rem; font-weight: 700; color: #0F172A; margin: 0;">
        🛡️ Autonomous Multi-Persona Response Playbooks
    </h2>
    <p style="font-size: 0.85rem; color: #64748B; margin: 2px 0 16px 0;">
        LangGraph orchestrated, rule-bounded directives customized across civic, institutional, and facilities operational domains.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

# Modern Horizontal Top Tabs (Full Container Width)
tab_commuter, tab_schools, tab_facilities, tab_municipal = st.tabs(
    [
        "🧑💼 Citizen & Commuter",
        "🏫 Schools & Campuses",
        "🏢 Facilities & HVAC",
        "🏛️ Municipal Mitigation",
    ]
)

commuter_data = playbook.get("commuter", {})
school_data = playbook.get("school", {})
facilities_data = playbook.get("facilities", {})
municipal_data = playbook.get("municipal", {})

with tab_commuter:
    mask_badge = commuter_data.get("mask_recommendation", "None Required")
    if "N95" in mask_badge:
        badge_bg, badge_color = "#FEF2F2", "#991B1B"
    elif "Surgical" in mask_badge or "3-Ply" in mask_badge:
        badge_bg, badge_color = "#FFFBEB", "#92400E"
    else:
        badge_bg, badge_color = "#ECFDF5", "#065F46"

    dep_adv = commuter_data.get(
        "departure_window_advice",
        "Standard commute timing; minimal atmospheric exposure risk.",
    )
    tra_adv = commuter_data.get(
        "transit_route_advice",
        "Standard transit choices; active walking and cycling are safe.",
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 🧑💼 Citizen & Commuter Protective Directives")
        with c2:
            render_html(
                f'<div style="text-align: right;"><span class="ag-badge"'
                f' style="background: {badge_bg}; color: {badge_color}; border:'
                f" 1px solid {badge_color}33; font-size: 0.88rem; padding: 6px"
                f' 14px;">😷 Mask Directive: {mask_badge}</span></div>'
            )

        st.markdown("---")
        render_html(f"""
        <div class="ag-directive-row">
            <div class="ag-directive-label">Optimal Travel & Departure Timing</div>
            <div class="ag-directive-value">{dep_adv}</div>
        </div>
        <div class="ag-directive-row">
            <div class="ag-directive-label">Transit Route & Modal Guidance</div>
            <div class="ag-directive-value">{tra_adv}</div>
        </div>
        """)

with tab_schools:
    recess_badge = school_data.get("recess_protocol", "Normal Schedule")
    if "Suspend" in recess_badge:
        rec_bg, rec_color = "#FEF2F2", "#991B1B"
    elif "Indoors" in recess_badge:
        rec_bg, rec_color = "#FFFBEB", "#92400E"
    else:
        rec_bg, rec_color = "#ECFDF5", "#065F46"

    vent_timing = school_data.get(
        "ventilation_window_timing",
        "Natural cross-ventilation encouraged during school hours.",
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown("### 🏫 Schools, Daycares & Campus Protocol")
        with c2:
            render_html(
                f'<div style="text-align: right;"><span class="ag-badge"'
                f' style="background: {rec_bg}; color: {rec_color}; border: 1px'
                f" solid {rec_color}33; font-size: 0.88rem; padding: 6px"
                f' 14px;">📋 Recess Directive: {recess_badge}</span></div>'
            )

        st.markdown("---")
        indoor_note = (
            school_data.get("recess_guidance")
            or (
                "PE and sports activities transitioned indoors or postponed."
                " Sensitive students monitored."
                if ("Suspend" in recess_badge or "Indoors" in recess_badge)
                else (
                    "Outdoor physical education and sports activities proceed on"
                    " normal operating schedules."
                )
            )
        )

        render_html(f"""
        <div class="ag-directive-row">
            <div class="ag-directive-label">Institutional Recess & Sports Guidance</div>
            <div class="ag-directive-value">{indoor_note}</div>
        </div>
        <div class="ag-directive-row">
            <div class="ag-directive-label">Classroom Window & Ventilation Schedule</div>
            <div class="ag-directive-value">{vent_timing}</div>
        </div>
        """)

with tab_facilities:
    recirc_badge = facilities_data.get("recirculation_mode", "100% Outdoor Air")
    scrub_mins = facilities_data.get("pre_scrub_minutes", 30)
    damper_advice = (
        facilities_data.get("damper_and_filtration_advice")
        or f"Configure fresh-air intake dampers in accordance with {recirc_badge}. Maintain proper static differential pressure across AHU filtration stages."
    )
    scrub_advice = (
        facilities_data.get("pre_scrub_advice")
        or f"Initiate building ventilation pre-scrub sequence {scrub_mins} minutes prior to core occupancy shifts to ensure target indoor air purity standards."
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                "### 🏢 Commercial Real Estate & Industrial HVAC Operations"
            )
        with c2:
            render_html(
                '<div style="text-align: right;"><span class="ag-badge"'
                ' style="background: #F0FDF4; color: #166534; border: 1px'
                " solid #BBF7D0; font-size: 0.85rem; padding: 6px 12px; margin-right:"
                f' 6px;">🔄 {recirc_badge}</span><span class="ag-badge"'
                ' style="background: #EFF6FF; color: #1E40AF; border: 1px'
                f' solid #BFDBFE; font-size: 0.85rem; padding: 6px 12px;">⏱️ {scrub_mins} Mins Pre-Scrub</span></div>'
            )

        st.markdown("---")
        render_html(f"""
        <div class="ag-directive-row">
            <div class="ag-directive-label">Damper & Air-Handling Unit (AHU) Settings</div>
            <div class="ag-directive-value">{damper_advice}</div>
        </div>
        <div class="ag-directive-row">
            <div class="ag-directive-label">Pre-Occupancy Purge / Scrub Sequence</div>
            <div class="ag-directive-value">{scrub_advice}</div>
        </div>
        """)

with tab_municipal:
    muni_action = municipal_data.get(
        "source_action", "Deploy Mechanical Sweepers Upwind"
    )
    upwind_bearing_card = diag_prof.get(
        "upwind_bearing_label", "Local Regional Sector"
    )
    civic_detail = (
        municipal_data.get("civic_action_detail")
        or f"Municipal pollution control enforcement units and squads to execute priority intervention: <b>{muni_action}</b>."
    )
    target_corridor = municipal_data.get(
        "target_corridor", f"Routine sweeping along {upwind_bearing_card}"
    )

    with st.container(border=True):
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                "### 🏛️ Municipal Mitigation & Source Enforcement Brief"
            )
        with c2:
            render_html(
                f'<div style="text-align: right;"><span class="ag-badge"'
                ' style="background: #FEF3C7; color: #92400E; border: 1px'
                f' solid #FDE68A; font-size: 0.88rem; padding: 6px 14px;">🚨 Priority Action: {muni_action}</span></div>'
            )

        st.markdown("---")
        render_html(f"""
        <div class="ag-directive-row">
            <div class="ag-directive-label">Civic Intervention Directive</div>
            <div class="ag-directive-value">{civic_detail}</div>
        </div>
        <div class="ag-directive-row">
            <div class="ag-directive-label">Geographic Target Corridor (Upwind Vector)</div>
            <div class="ag-directive-value"><b>Target Source Bearing:</b> {target_corridor} (Air mass advecting from {upwind_bearing_card})</div>
        </div>
        """)


# Collapsible Section: Chronological Agent Execution Trace Log
trace_steps = agent_state.get("trace", [])
exec_ms = agent_state.get("execution_time_ms", 0)

with st.expander(
    f"🔍 Agent Execution Audit Log ({len(trace_steps)} Steps | {exec_ms} ms)",
    expanded=False,
):
    st.caption(
        "Chronological audit trace detailing telemetry ingestion, River online"
        " learning fit, Sentinel check, Diagnostic matrix, and Amazon Bedrock (Nova Pro)"
        " reasoning via AWS Strands Agents SDK."
    )
    st.code("\n".join(trace_steps), language="bash")


# Footer
render_html("""
<div style="margin-top: 36px; padding-top: 16px; border-top: 1px solid #E2E8F0; text-align: center; font-size: 0.78rem; color: #94A3B8;">
    AeroGraph &middot; Autonomous Air Quality Orchestrator &middot; River Streaming ML &middot; Amazon Bedrock (Nova Pro) &middot; AWS Strands Agents SDK &middot; Open-Meteo &middot; Asia/Kolkata (IST)
</div>
""")
