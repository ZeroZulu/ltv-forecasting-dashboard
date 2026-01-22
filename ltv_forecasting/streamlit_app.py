"""
LTV Forecasting Dashboard - Streamlit App (FIXED)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="LTV Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0a0a0c;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111113;
    }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1d 0%, #111113 100%);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 24px;
        margin: 8px 0;
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #22c55e;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-label {
        font-size: 12px;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Table styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        background: #111113;
        border-radius: 12px;
        overflow: hidden;
    }
    
    .styled-table th {
        background: #1a1a1d;
        color: #888;
        padding: 16px;
        text-align: left;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .styled-table td {
        padding: 16px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        color: #fff;
    }
    
    .styled-table tr:hover {
        background: rgba(34, 197, 94, 0.05);
    }
    
    /* Status badges */
    .status-badge {
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    
    .status-excellent {
        background: rgba(34, 197, 94, 0.2);
        color: #22c55e;
    }
    
    .status-good {
        background: rgba(59, 130, 246, 0.2);
        color: #3b82f6;
    }
    
    .status-warning {
        background: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Generate sample data
@st.cache_data
def load_data():
    np.random.seed(42)
    
    # Channel data
    channels = pd.DataFrame({
        'channel': ['Referral', 'Organic Search', 'Cross Promo', 'Influencer', 'Paid Social', 'App Store'],
        'players': [1526, 2035, 961, 1479, 2547, 1452],
        'avg_ltv': [714, 518, 499, 455, 429, 341],
        'cac': [5.0, 0.0, 3.0, 25.0, 12.5, 8.0],
        'roi': [143, 99999, 166, 18, 34, 43],
        'retention': [24, 25, 23, 22, 21, 20]
    })
    
    # Monthly LTV data
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
    monthly_ltv = pd.DataFrame({
        'month': months,
        'ltv': [180000, 210000, 250000, 320000, 380000, 520000, 620000, 580000, 540000]
    })
    
    # Player segments
    segments = pd.DataFrame({
        'segment': ['Whale', 'Dolphin', 'Minnow', 'F2P'],
        'count': [500, 800, 2700, 6000],
        'revenue_pct': [85, 10, 4, 1]
    })
    
    return channels, monthly_ltv, segments

channels_df, monthly_df, segments_df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("### 🎮 LTV Analytics")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["Dashboard", "Channels", "Segments", "Model Performance"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### Filters")
    
    date_range = st.selectbox("Time Period", ["Last 30 days", "Last 90 days", "Last 12 months", "All time"])
    
    selected_channels = st.multiselect(
        "Channels",
        channels_df['channel'].tolist(),
        default=channels_df['channel'].tolist()
    )

# Main content
if page == "Dashboard":
    st.title("📊 Dashboard")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">$4.88M</div>
            <div style="color: #22c55e; font-size: 14px;">↑ 12.3% vs last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Average LTV</div>
            <div class="metric-value">$488</div>
            <div style="color: #22c55e; font-size: 14px;">↑ 8.7% vs baseline</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model R²</div>
            <div class="metric-value">0.891</div>
            <div style="color: #888; font-size: 14px;">Industry: 0.60-0.75</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Players</div>
            <div class="metric-value">10,000</div>
            <div style="color: #22c55e; font-size: 14px;">6 ML Models</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 LTV Trend")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly_df['month'],
            y=monthly_df['ltv'],
            marker_color='#22c55e',
            opacity=0.8
        ))
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Channel Distribution")
        fig = px.pie(
            channels_df,
            values='players',
            names='channel',
            color_discrete_sequence=['#22c55e', '#4ade80', '#86efac', '#3b82f6', '#60a5fa', '#93c5fd']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Performance Table
    st.subheader("📊 Channel Performance")
    
    # Use Streamlit's native dataframe display with custom formatting
    display_df = channels_df[channels_df['channel'].isin(selected_channels)].copy()
    display_df['ROI'] = display_df['roi'].apply(lambda x: f"{x}x" if x < 1000 else "∞")
    display_df['Avg LTV'] = display_df['avg_ltv'].apply(lambda x: f"${x:,}")
    display_df['CAC'] = display_df['cac'].apply(lambda x: f"${x:.2f}")
    display_df['Players'] = display_df['players'].apply(lambda x: f"{x:,}")
    
    # Status column
    def get_status(roi):
        if roi > 100:
            return "🟢 Excellent"
        elif roi > 30:
            return "🔵 Good"
        else:
            return "🟡 Optimize"
    
    display_df['Status'] = display_df['roi'].apply(get_status)
    
    st.dataframe(
        display_df[['channel', 'Players', 'Avg LTV', 'CAC', 'ROI', 'Status']].rename(columns={
            'channel': 'Channel'
        }),
        use_container_width=True,
        hide_index=True
    )

elif page == "Channels":
    st.title("📡 Channel Analysis")
    
    # Channel comparison chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Avg LTV',
        x=channels_df['channel'],
        y=channels_df['avg_ltv'],
        marker_color='#22c55e'
    ))
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#888',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='Average LTV ($)'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.subheader("💰 ROI Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI by channel (excluding infinite)
        roi_df = channels_df[channels_df['roi'] < 1000].copy()
        fig = px.bar(
            roi_df,
            x='channel',
            y='roi',
            color='roi',
            color_continuous_scale=['#f59e0b', '#22c55e'],
            title='LTV:CAC Ratio by Channel'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Player distribution
        fig = px.bar(
            channels_df,
            x='channel',
            y='players',
            color='channel',
            color_discrete_sequence=['#22c55e', '#4ade80', '#86efac', '#3b82f6', '#60a5fa', '#93c5fd'],
            title='Players by Channel'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            showlegend=False,
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)

elif page == "Segments":
    st.title("👥 Player Segments")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Player Distribution")
        fig = px.pie(
            segments_df,
            values='count',
            names='segment',
            color_discrete_sequence=['#22c55e', '#4ade80', '#86efac', '#d1d5db']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Revenue Contribution")
        fig = px.bar(
            segments_df,
            x='segment',
            y='revenue_pct',
            color='segment',
            color_discrete_sequence=['#22c55e', '#4ade80', '#86efac', '#d1d5db'],
            title='% of Total Revenue'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            showlegend=False,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    st.info("💡 **Key Insight:** Whales (5% of players) generate 85% of total revenue!")

elif page == "Model Performance":
    st.title("🤖 Model Performance")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", "0.891", "+0.14 vs baseline")
    with col2:
        st.metric("MAE", "$53.23", "")
    with col3:
        st.metric("RMSE", "$213.97", "")
    with col4:
        st.metric("Cross-Val R²", "0.891 ± 0.057", "5-fold CV")
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("🔑 Feature Importance")
    
    features = pd.DataFrame({
        'feature': ['early_spend', 'early_avg_purchase', 'early_purchases', 'tenure_days', 
                   'channel_referral', 'channel_organic', 'channel_paid_social'],
        'importance': [0.898, 0.078, 0.010, 0.005, 0.004, 0.003, 0.002]
    })
    
    fig = px.bar(
        features,
        x='importance',
        y='feature',
        orientation='h',
        color='importance',
        color_continuous_scale=['#3b82f6', '#22c55e']
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#888',
        yaxis=dict(autorange='reversed'),
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model info
    st.subheader("📋 Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Model Type:** Gradient Boosting Regressor
        
        **Hyperparameters:**
        - n_estimators: 100
        - max_depth: 4
        - learning_rate: 0.1
        - min_samples_leaf: 20
        """)
    
    with col2:
        st.markdown("""
        **Key Improvements (v3):**
        - ✅ Removed feature leakage
        - ✅ Added regularization
        - ✅ Cross-validation
        - ✅ Early behavior signals only
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 12px;'>"
    "LTV Forecasting Dashboard | Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True
)
