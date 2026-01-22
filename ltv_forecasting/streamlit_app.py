"""
LTV Forecasting Dashboard - Skeuomorphic Design
Executive-Level Marketing Analytics Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="LTV Analytics | Marketing Intelligence",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - SKEUOMORPHIC DESIGN
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(145deg, #0a0a0a 0%, #111111 50%, #0d0d0d 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: linear-gradient(145deg, rgba(20, 20, 20, 0.9) 0%, rgba(15, 15, 15, 0.95) 100%);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 
            0 4px 24px rgba(0, 0, 0, 0.4),
            0 1px 2px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(145deg, #141414 0%, #0f0f0f 100%);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.5),
            0 2px 8px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.02);
        position: relative;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 12px 40px rgba(0, 0, 0, 0.6),
            0 4px 12px rgba(0, 0, 0, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    
    .metric-label {
        font-size: 12px;
        font-weight: 500;
        color: #6b6b6b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: -0.5px;
    }
    
    .metric-subtext {
        font-size: 12px;
        color: #4a4a4a;
        margin-top: 8px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .metric-subtext.positive {
        color: #22c55e;
    }
    
    /* Green Accent Elements */
    .accent-green {
        color: #22c55e;
    }
    
    .accent-bar {
        height: 3px;
        background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
        border-radius: 2px;
        margin-top: 12px;
    }
    
    .accent-dot {
        width: 8px;
        height: 8px;
        background: #22c55e;
        border-radius: 50%;
        box-shadow: 0 0 8px rgba(34, 197, 94, 0.5);
        display: inline-block;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 14px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Credit Card Style Widget */
    .card-widget {
        background: linear-gradient(135deg, #1a1a1a 0%, #0f0f0f 50%, #1a1a1a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 10px 40px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .card-widget::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 100%;
        height: 100%;
        background: radial-gradient(circle, rgba(34, 197, 94, 0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    
    /* Progress Bars */
    .progress-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
        height: 8px;
        overflow: hidden;
        margin: 8px 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }
    
    .progress-green { background: linear-gradient(90deg, #22c55e, #16a34a); }
    .progress-blue { background: linear-gradient(90deg, #3b82f6, #2563eb); }
    .progress-purple { background: linear-gradient(90deg, #8b5cf6, #7c3aed); }
    .progress-orange { background: linear-gradient(90deg, #f59e0b, #d97706); }
    
    /* Data Table */
    .data-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .data-table th {
        text-align: left;
        font-size: 11px;
        font-weight: 600;
        color: #5a5a5a;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 12px 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .data-table td {
        padding: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03);
        color: #e0e0e0;
        font-size: 14px;
    }
    
    .data-table tr:hover td {
        background: rgba(255, 255, 255, 0.02);
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 500;
    }
    
    .status-excellent {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .status-good {
        background: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
    
    .status-moderate {
        background: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    
    /* Button Styles */
    .btn-primary {
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        color: #000;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
        transition: all 0.2s ease;
    }
    
    .btn-secondary {
        background: rgba(255, 255, 255, 0.05);
        color: #fff;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    /* Nav Items */
    .nav-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 16px;
        border-radius: 10px;
        color: #6b6b6b;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-bottom: 4px;
    }
    
    .nav-item:hover {
        background: rgba(255, 255, 255, 0.03);
        color: #ffffff;
    }
    
    .nav-item.active {
        background: rgba(34, 197, 94, 0.1);
        color: #22c55e;
        border: 1px solid rgba(34, 197, 94, 0.2);
    }
    
    .nav-icon {
        width: 20px;
        height: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    /* Logo */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 8px 16px;
        margin-bottom: 24px;
    }
    
    .logo-icon {
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.3);
    }
    
    .logo-text {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    /* Search Box */
    .search-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 10px 14px;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 24px;
    }
    
    .search-box input {
        background: transparent;
        border: none;
        color: #6b6b6b;
        font-size: 13px;
        width: 100%;
        outline: none;
    }
    
    /* Chart container */
    .chart-container {
        background: linear-gradient(145deg, #141414 0%, #0f0f0f 100%);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    }
    
    /* Streamlit overrides */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        padding: 10px 20px;
        color: #6b6b6b;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(34, 197, 94, 0.15);
        color: #22c55e;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: rgba(255, 255, 255, 0.05);
        margin: 20px 0;
    }
    
    /* Feature sections */
    .features-label {
        font-size: 11px;
        font-weight: 600;
        color: #4a4a4a;
        text-transform: uppercase;
        letter-spacing: 1px;
        padding: 0 16px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_data():
    """Load all data files."""
    base_path = Path(__file__).parent
    data_paths = [base_path / "data", base_path / "outputs", Path("data"), Path("outputs"), Path(".")]
    
    data = {}
    for path in data_paths:
        try:
            if (path / "dashboard_data.json").exists():
                with open(path / "dashboard_data.json", "r") as f:
                    data['dashboard'] = json.load(f)
            if (path / "player_predictions.csv").exists():
                data['predictions'] = pd.read_csv(path / "player_predictions.csv")
            if (path / "channel_insights.csv").exists():
                data['channels'] = pd.read_csv(path / "channel_insights.csv")
            if (path / "feature_importance.csv").exists():
                data['features'] = pd.read_csv(path / "feature_importance.csv")
            if data:
                break
        except Exception:
            continue
    
    if not data:
        data = generate_sample_data()
    return data


def generate_sample_data():
    """Generate sample data for demo."""
    np.random.seed(42)
    channels = ['referral', 'organic_search', 'cross_promo', 'influencer', 'paid_social', 'app_store']
    
    channel_data = pd.DataFrame({
        'acquisition_channel': channels,
        'avg_ltv': [714.00, 517.60, 498.66, 455.12, 428.92, 341.47],
        'median_ltv': [2.43, 2.27, 2.38, 2.43, 2.28, 2.34],
        'ltv_std': [2733.62, 1978.61, 1919.30, 1768.48, 1652.65, 1299.16],
        'player_count': [1526, 2035, 961, 1479, 2547, 1452],
        'avg_retention_prob': [0.24, 0.25, 0.24, 0.24, 0.24, 0.24],
        'avg_predicted_purchases': [1.01, 0.86, 0.85, 0.80, 0.76, 0.67],
        'avg_transaction_value': [40.62, 39.17, 39.42, 38.42, 39.26, 37.83],
        'ltv_10th_percentile': [635.59, 540.12, 532.34, 491.60, 500.62, 442.37],
        'ltv_90th_percentile': [1380.59, 1219.22, 1199.42, 1114.30, 1178.29, 1080.40],
        'avg_cac': [5.0, 0.01, 3.0, 25.0, 12.5, 8.0],
        'ltv_cac_ratio': [142.80, 51760.0, 166.22, 18.20, 34.31, 42.68]
    })
    
    features = pd.DataFrame({
        'feature': ['frequency', 'monetary_mean', 'monetary_max', 'monetary_std', 
                   'purchase_velocity', 'early_7d_spend', 'customer_lifespan',
                   'total_playtime', 'total_sessions', 'engagement_consistency'],
        'importance': [0.778, 0.208, 0.013, 0.002, 0.0002, 0.0001, 0.00003, 0.00002, 0.00002, 0.00001]
    })
    
    dashboard = {
        'metrics': {'r2': 0.825, 'mae': 328.30, 'rmse': 1355.22, 'mape': 20.3},
        'total_players': 10000,
        'total_predicted_ltv': 4780000,
        'avg_ltv': 478
    }
    
    n = 1000
    predictions = pd.DataFrame({
        'player_id': [f'P{str(i).zfill(6)}' for i in range(n)],
        'acquisition_channel': np.random.choice(channels, n),
        'ltv_ensemble': np.random.exponential(500, n),
        'alive_probability': np.random.beta(2, 6, n),
        'predicted_purchases': np.random.poisson(2, n),
    })
    
    return {'dashboard': dashboard, 'channels': channel_data, 'features': features, 'predictions': predictions}


# ============================================================================
# CHART FUNCTIONS
# ============================================================================

def create_bar_chart(channel_data):
    """Create monthly revenue style bar chart."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    values = [340, 380, 320, 410, 680, 520, 480, 450, 400]
    
    fig = go.Figure()
    
    colors = ['rgba(34, 197, 94, 0.3)' if i != 4 else '#22c55e' for i in range(len(months))]
    
    fig.add_trace(go.Bar(
        x=months,
        y=values,
        marker_color=colors,
        marker_line_width=0,
        hovertemplate='%{x}: $%{y}K<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#6b6b6b', size=11),
        height=200,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickprefix='$',
            ticksuffix='k',
            tickfont=dict(size=10)
        ),
        bargap=0.5,
        showlegend=False
    )
    
    return fig


def create_channel_performance_chart(channel_data):
    """Create channel LTV horizontal bar chart."""
    df = channel_data.sort_values('avg_ltv', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df['acquisition_channel'].str.replace('_', ' ').str.title(),
        x=df['avg_ltv'],
        orientation='h',
        marker=dict(
            color=['#22c55e' if i == len(df)-1 else 'rgba(34, 197, 94, 0.3)' 
                   for i in range(len(df))],
            line_width=0
        ),
        text=[f'${v:,.0f}' for v in df['avg_ltv']],
        textposition='outside',
        textfont=dict(color='#22c55e', size=11),
        hovertemplate='<b>%{y}</b><br>Avg LTV: $%{x:,.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#6b6b6b'),
        height=280,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickformat='$,.0f',
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickfont=dict(size=11, color='#a0a0a0')
        ),
        showlegend=False
    )
    
    return fig


def create_cohort_lines(channel_data):
    """Create cohort LTV lines."""
    months = list(range(1, 13))
    
    fig = go.Figure()
    
    colors = {'referral': '#22c55e', 'organic_search': '#3b82f6', 
              'paid_social': '#8b5cf6', 'cross_promo': '#f59e0b'}
    
    for _, row in channel_data.head(4).iterrows():
        channel = row['acquisition_channel']
        ltv = row['avg_ltv']
        values = [ltv * (m / 12) ** 0.7 for m in months]
        
        fig.add_trace(go.Scatter(
            x=months,
            y=values,
            mode='lines',
            name=channel.replace('_', ' ').title(),
            line=dict(color=colors.get(channel, '#22c55e'), width=2),
            hovertemplate='Month %{x}: $%{y:.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#6b6b6b', size=11),
        height=250,
        margin=dict(l=0, r=0, t=10, b=30),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickfont=dict(size=10),
            dtick=2
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.03)',
            showline=False,
            tickprefix='$',
            tickfont=dict(size=10)
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        showlegend=True
    )
    
    return fig


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    data = load_data()
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-container">
            <div class="logo-icon">🎮</div>
            <div class="logo-text">LTV Analytics</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Search
        st.markdown("""
        <div class="search-box">
            <span style="color: #4a4a4a;">🔍</span>
            <input type="text" placeholder="Search Anything..." disabled>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation - Menu
        st.markdown('<div class="features-label">MENU</div>', unsafe_allow_html=True)
        
        menu_items = [
            ('📊', 'Dashboard', True),
            ('🔔', 'Notifications', False),
            ('📈', 'Analytics', False),
            ('💳', 'Transactions', False),
            ('🎴', 'Channels', False),
            ('📜', 'History', False),
        ]
        
        for icon, label, active in menu_items:
            active_class = 'active' if active else ''
            st.markdown(f"""
            <div class="nav-item {active_class}">
                <span class="nav-icon">{icon}</span>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Features
        st.markdown('<div class="features-label">FEATURES</div>', unsafe_allow_html=True)
        
        feature_items = [('🔗', 'Integration'), ('⚡', 'Automation')]
        for icon, label in feature_items:
            st.markdown(f"""
            <div class="nav-item">
                <span class="nav-icon">{icon}</span>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Tools
        st.markdown('<div class="features-label">TOOLS</div>', unsafe_allow_html=True)
        
        tool_items = [('⚙️', 'Settings'), ('❓', 'Help Center')]
        for icon, label in tool_items:
            st.markdown(f"""
            <div class="nav-item">
                <span class="nav-icon">{icon}</span>
                <span>{label}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        
        # Upgrade Card
        st.markdown("""
        <div class="glass-card" style="margin-top: 20px; padding: 20px;">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">
                <span style="color: #22c55e; font-size: 18px;">⚡</span>
                <span style="font-weight: 600; color: #fff;">Upgrade Pro!</span>
            </div>
            <p style="color: #6b6b6b; font-size: 12px; margin-bottom: 16px;">
                Unlock advanced LTV models and real-time predictions
            </p>
            <div style="display: flex; gap: 8px;">
                <button class="btn-primary" style="flex: 1;">Upgrade</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== MAIN CONTENT ==========
    
    # Header
    col_header1, col_header2 = st.columns([3, 1])
    with col_header1:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
            <h1 style="font-size: 24px; font-weight: 600; color: #fff; margin: 0;">Dashboard</h1>
            <div style="display: flex; gap: 12px;">
                <button class="btn-secondary">📄 Generate Report</button>
                <button class="btn-secondary">⬇️ Export</button>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_header2:
        st.markdown("""
        <div class="search-box" style="margin-bottom: 0;">
            <span style="color: #4a4a4a;">🔍</span>
            <input type="text" placeholder="Search Anything..." disabled>
        </div>
        """, unsafe_allow_html=True)
    
    # Top KPI Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue This Month</div>
            <div class="metric-value" style="color: #22c55e;">${data['dashboard']['total_predicted_ltv']:,.0f}</div>
            <div class="metric-subtext positive">
                <span class="accent-dot"></span>
                Vs Last month
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average Player LTV</div>
            <div class="metric-value">${data['dashboard']['avg_ltv']:,.2f}</div>
            <div class="metric-subtext positive">
                <span class="accent-dot"></span>
                +12.3% vs baseline
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy (R²)</div>
            <div class="metric-value" style="color: #22c55e;">{data['dashboard']['metrics']['r2']:.1%}</div>
            <div class="metric-subtext positive">
                <span class="accent-dot"></span>
                Industry: 60-75%
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Middle Row - Cards and Chart
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        # Model Performance Card (like credit card widget)
        st.markdown("""
        <div class="card-widget">
            <div class="section-header">
                <span>🧠</span>
                <span>AI-Generated Predictions</span>
            </div>
            <div style="margin-bottom: 20px;">
                <div style="font-size: 28px; font-weight: 700; color: #22c55e; font-family: 'JetBrains Mono';">
                    10,000 Players
                </div>
                <div style="color: #6b6b6b; font-size: 12px;">Analyzed with 6 ML Models</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Channel distribution bars
        channel_dist = [
            ('Paid Social', 27, 'green'),
            ('Organic', 35, 'blue'),
            ('Referral', 18, 'purple'),
            ('Other', 20, 'orange'),
        ]
        
        st.markdown('<div style="margin-top: 16px;">', unsafe_allow_html=True)
        for name, pct, color in channel_dist:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <div style="width: 8px; height: 8px; border-radius: 2px; background: var(--{color}-color, #22c55e);"></div>
                <span style="font-size: 11px; color: #6b6b6b; width: 80px;">{name} ({pct}%)</span>
                <div style="flex: 1; height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px;">
                    <div style="width: {pct}%; height: 100%; background: linear-gradient(90deg, #22c55e, #16a34a); border-radius: 2px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col_right:
        # Revenue Chart
        st.markdown("""
        <div class="chart-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                <div>
                    <div class="metric-label">Available LTV Pool</div>
                    <div class="metric-value">$4,780,000</div>
                </div>
                <div style="display: flex; gap: 8px;">
                    <button class="btn-secondary" style="padding: 6px 12px; font-size: 12px;">Line view</button>
                    <button class="btn-primary" style="padding: 6px 12px; font-size: 12px;">Bar view</button>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(create_bar_chart(data['channels']), use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("""
            <div class="metric-subtext positive" style="margin-top: 8px;">
                <span class="accent-dot"></span>
                Vs Last month
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Channel Performance Table
    st.markdown("""
    <div class="chart-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <div class="search-box" style="width: 250px; margin: 0;">
                <span style="color: #4a4a4a;">🔍</span>
                <input type="text" placeholder="Search Channel..." disabled>
            </div>
            <div style="display: flex; gap: 12px;">
                <button class="btn-secondary" style="padding: 8px 16px; font-size: 12px;">📥 Import</button>
                <button class="btn-primary" style="padding: 8px 16px; font-size: 12px;">📤 Export</button>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Build table HTML
    table_html = """
    <table class="data-table">
        <thead>
            <tr>
                <th>Channel</th>
                <th>Players</th>
                <th>Avg LTV</th>
                <th>CAC</th>
                <th>ROI</th>
                <th>Retention</th>
                <th>Status</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for _, row in data['channels'].iterrows():
        roi = row['ltv_cac_ratio']
        status_class = 'excellent' if roi > 100 else 'good' if roi > 30 else 'moderate'
        status_text = 'Excellent' if roi > 100 else 'Good' if roi > 30 else 'Moderate'
        
        channel_icons = {
            'paid_social': '📱',
            'organic_search': '🔍', 
            'referral': '🤝',
            'influencer': '⭐',
            'app_store': '📲',
            'cross_promo': '🔄'
        }
        icon = channel_icons.get(row['acquisition_channel'], '📊')
        
        table_html += f"""
        <tr>
            <td>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 32px; height: 32px; background: rgba(34, 197, 94, 0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center;">{icon}</div>
                    <span style="font-weight: 500;">{row['acquisition_channel'].replace('_', ' ').title()}</span>
                </div>
            </td>
            <td>{row['player_count']:,}</td>
            <td style="font-family: 'JetBrains Mono'; color: #22c55e; font-weight: 600;">${row['avg_ltv']:,.0f}</td>
            <td style="font-family: 'JetBrains Mono';">${row['avg_cac']:.2f}</td>
            <td style="font-family: 'JetBrains Mono';">{roi:.0f}x</td>
            <td>{row['avg_retention_prob']:.0%}</td>
            <td><span class="status-badge status-{status_class}">{status_text}</span></td>
        </tr>
        """
    
    table_html += """
        </tbody>
    </table>
    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; padding-top: 16px; border-top: 1px solid rgba(255,255,255,0.05);">
        <div style="display: flex; gap: 8px;">
            <span style="color: #4a4a4a; font-size: 12px;">Page</span>
            <button style="background: rgba(34, 197, 94, 0.15); color: #22c55e; border: none; padding: 4px 10px; border-radius: 4px; font-size: 12px;">1</button>
            <button style="background: transparent; color: #6b6b6b; border: none; padding: 4px 10px; font-size: 12px;">2</button>
            <button style="background: transparent; color: #6b6b6b; border: none; padding: 4px 10px; font-size: 12px;">3</button>
        </div>
        <span style="color: #4a4a4a; font-size: 12px;">Showing 1 to 6 of 6 entries</span>
    </div>
    </div>
    """
    
    st.markdown(table_html, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Bottom Row - Feature Importance & Cohort
    col_feat, col_cohort = st.columns(2)
    
    with col_feat:
        st.markdown("""
        <div class="chart-container">
            <div class="section-header">
                <span>🎯</span>
                <span>Top Predictive Features</span>
            </div>
        """, unsafe_allow_html=True)
        
        for i, row in data['features'].head(6).iterrows():
            pct = row['importance'] * 100
            width = min(pct / data['features']['importance'].max() * 100, 100)
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                <div style="width: 24px; height: 24px; background: rgba(34, 197, 94, 0.1); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: #6b6b6b;">{i+1}</div>
                <span style="width: 120px; font-size: 12px; color: #a0a0a0;">{row['feature'].replace('_', ' ').title()}</span>
                <div style="flex: 1; height: 6px; background: rgba(255,255,255,0.05); border-radius: 3px;">
                    <div style="width: {width}%; height: 100%; background: linear-gradient(90deg, #22c55e, #16a34a); border-radius: 3px;"></div>
                </div>
                <span style="width: 50px; text-align: right; font-size: 11px; color: #22c55e; font-family: 'JetBrains Mono';">{pct:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_cohort:
        st.markdown("""
        <div class="chart-container">
            <div class="section-header">
                <span>📈</span>
                <span>LTV Cohort Analysis</span>
            </div>
        """, unsafe_allow_html=True)
        
        st.plotly_chart(create_cohort_lines(data['channels']), use_container_width=True, config={'displayModeBar': False})
        
        st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
