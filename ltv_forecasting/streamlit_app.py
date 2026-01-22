"""
LTV Forecasting Dashboard - Streamlit App (with Working Filters)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="LTV Analytics Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0a0a0c; }
    [data-testid="stSidebar"] { background-color: #111113; }
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
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Data generation functions - these respond to filters
@st.cache_data
def generate_base_data():
    """Generate base channel data."""
    np.random.seed(42)
    return pd.DataFrame({
        'channel': ['Referral', 'Organic Search', 'Cross Promo', 'Influencer', 'Paid Social', 'App Store'],
        'players': [1526, 2035, 961, 1479, 2547, 1452],
        'avg_ltv': [714, 518, 499, 455, 429, 341],
        'cac': [5.0, 0.0, 3.0, 25.0, 12.5, 8.0],
        'roi': [143, 99999, 166, 18, 34, 43],
        'retention': [24, 25, 23, 22, 21, 20]
    })

def get_filtered_data(time_period, selected_channels):
    """Generate data based on selected time period."""
    base_df = generate_base_data()
    
    # Apply channel filter
    filtered_df = base_df[base_df['channel'].isin(selected_channels)].copy()
    
    # Apply time period multipliers (simulating different time windows)
    multipliers = {
        'Last 30 days': {'ltv': 0.3, 'players': 0.25, 'trend_months': 3},
        'Last 90 days': {'ltv': 0.6, 'players': 0.5, 'trend_months': 5},
        'Last 12 months': {'ltv': 1.0, 'players': 1.0, 'trend_months': 9},
        'All time': {'ltv': 1.2, 'players': 1.15, 'trend_months': 12}
    }
    
    mult = multipliers.get(time_period, multipliers['Last 12 months'])
    
    # Adjust values based on time period
    filtered_df['avg_ltv'] = (filtered_df['avg_ltv'] * mult['ltv']).astype(int)
    filtered_df['players'] = (filtered_df['players'] * mult['players']).astype(int)
    
    # Recalculate ROI
    filtered_df['roi'] = filtered_df.apply(
        lambda x: int(x['avg_ltv'] / x['cac']) if x['cac'] > 0 else 99999, axis=1
    )
    
    return filtered_df, mult['trend_months']

def get_monthly_data(num_months, selected_channels, base_df):
    """Generate monthly trend data."""
    all_months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = all_months[:num_months]
    
    # Calculate total players from selected channels
    total_players = base_df['players'].sum()
    
    # Generate trend with growth
    base_values = np.array([180, 210, 250, 320, 380, 520, 620, 580, 540, 500, 480, 520][:num_months])
    scale_factor = total_players / 10000  # Scale based on selected channels
    
    return pd.DataFrame({
        'month': months,
        'ltv': (base_values * scale_factor * 1000).astype(int)
    })

def get_segment_data(time_period):
    """Get segment data based on time period."""
    base_segments = {
        'Last 30 days': {'Whale': 120, 'Dolphin': 200, 'Minnow': 700, 'F2P': 1500},
        'Last 90 days': {'Whale': 280, 'Dolphin': 450, 'Minnow': 1500, 'F2P': 3500},
        'Last 12 months': {'Whale': 500, 'Dolphin': 800, 'Minnow': 2700, 'F2P': 6000},
        'All time': {'Whale': 580, 'Dolphin': 920, 'Minnow': 3100, 'F2P': 6900}
    }
    
    data = base_segments.get(time_period, base_segments['Last 12 months'])
    
    return pd.DataFrame({
        'segment': list(data.keys()),
        'count': list(data.values()),
        'revenue_pct': [85, 10, 4, 1]
    })

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
    
    time_period = st.selectbox(
        "Time Period",
        ["Last 30 days", "Last 90 days", "Last 12 months", "All time"],
        index=2  # Default to "Last 12 months"
    )
    
    base_channels = generate_base_data()
    selected_channels = st.multiselect(
        "Channels",
        base_channels['channel'].tolist(),
        default=base_channels['channel'].tolist()
    )
    
    # Show filter status
    st.markdown("---")
    st.markdown(f"**Active Filters:**")
    st.markdown(f"📅 {time_period}")
    st.markdown(f"📡 {len(selected_channels)} channels")

# Get filtered data
if len(selected_channels) == 0:
    st.warning("Please select at least one channel")
    st.stop()

channels_df, num_months = get_filtered_data(time_period, selected_channels)
monthly_df = get_monthly_data(num_months, selected_channels, channels_df)
segments_df = get_segment_data(time_period)

# Calculate summary metrics
total_revenue = channels_df['avg_ltv'].sum() * channels_df['players'].sum() // 1000
avg_ltv = int(channels_df['avg_ltv'].mean())
total_players = channels_df['players'].sum()

# Main content
if page == "Dashboard":
    st.title("📊 Dashboard")
    
    # Show active filter indicator
    st.caption(f"📅 Showing data for: **{time_period}** | 📡 {len(selected_channels)} channels selected")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Revenue</div>
            <div class="metric-value">${total_revenue:,}K</div>
            <div style="color: #22c55e; font-size: 14px;">↑ 12.3% vs last period</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Average LTV</div>
            <div class="metric-value">${avg_ltv}</div>
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
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Players</div>
            <div class="metric-value">{total_players:,}</div>
            <div style="color: #22c55e; font-size: 14px;">{len(selected_channels)} channels</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📈 LTV Trend ({time_period})")
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
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', title='LTV ($)'),
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
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Channel Performance Table
    st.subheader("📊 Channel Performance")
    
    display_df = channels_df.copy()
    display_df['ROI'] = display_df['roi'].apply(lambda x: f"{x}x" if x < 1000 else "∞")
    display_df['Avg LTV'] = display_df['avg_ltv'].apply(lambda x: f"${x:,}")
    display_df['CAC'] = display_df['cac'].apply(lambda x: f"${x:.2f}")
    display_df['Players'] = display_df['players'].apply(lambda x: f"{x:,}")
    
    def get_status(roi):
        if roi > 100:
            return "🟢 Excellent"
        elif roi > 30:
            return "🔵 Good"
        else:
            return "🟡 Optimize"
    
    display_df['Status'] = display_df['roi'].apply(get_status)
    
    st.dataframe(
        display_df[['channel', 'Players', 'Avg LTV', 'CAC', 'ROI', 'Status']].rename(columns={'channel': 'Channel'}),
        use_container_width=True,
        hide_index=True
    )

elif page == "Channels":
    st.title("📡 Channel Analysis")
    st.caption(f"📅 {time_period} | 📡 {len(selected_channels)} channels")
    
    # Channel comparison
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
        height=400,
        title=f'Average LTV by Channel ({time_period})'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        roi_df = channels_df[channels_df['roi'] < 1000].copy()
        fig = px.bar(
            roi_df,
            x='channel',
            y='roi',
            color='roi',
            color_continuous_scale=['#f59e0b', '#22c55e'],
            title='LTV:CAC Ratio'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
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
    st.caption(f"📅 {time_period}")
    
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
            color_discrete_sequence=['#22c55e', '#4ade80', '#86efac', '#d1d5db']
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#888',
            showlegend=False,
            height=400,
            yaxis_title='% of Revenue'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Segment table
    st.subheader("📊 Segment Details")
    seg_display = segments_df.copy()
    seg_display['Players'] = seg_display['count'].apply(lambda x: f"{x:,}")
    seg_display['Revenue %'] = seg_display['revenue_pct'].apply(lambda x: f"{x}%")
    st.dataframe(
        seg_display[['segment', 'Players', 'Revenue %']].rename(columns={'segment': 'Segment'}),
        use_container_width=True,
        hide_index=True
    )
    
    st.info(f"💡 **Key Insight ({time_period}):** Whales ({segments_df[segments_df['segment']=='Whale']['count'].values[0]:,} players) generate 85% of revenue!")

elif page == "Model Performance":
    st.title("🤖 Model Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("R² Score", "0.891", "+0.14 vs baseline")
    with col2:
        st.metric("MAE", "$53.23")
    with col3:
        st.metric("RMSE", "$213.97")
    with col4:
        st.metric("Cross-Val R²", "0.891 ± 0.057")
    
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
        - ✅ No feature leakage
        - ✅ Regularization added
        - ✅ 5-fold cross-validation
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
