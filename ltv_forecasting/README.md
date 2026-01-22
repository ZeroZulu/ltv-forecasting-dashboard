# 🎮 LTV Forecasting by Marketing Channel

## Executive-Level Marketing Analytics Platform

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Model R²](https://img.shields.io/badge/Model%20R²-82.5%25-success.svg)]()

**🔗 [Live Demo on Streamlit Cloud](https://ltv-forecasting.streamlit.app)** *(Update with your URL)*

**Advanced lifetime value prediction models linking acquisition channels and early player behavior to long-term value.** This project demonstrates production-grade ML techniques for customer analytics in the gaming/mobile industry.

![Dashboard Preview](https://via.placeholder.com/800x400?text=LTV+Dashboard+Preview)

---

## 📊 Project Overview

This project implements a comprehensive LTV (Lifetime Value) forecasting system that predicts player value across different marketing acquisition channels. The ensemble approach combines probabilistic models, survival analysis, and machine learning to provide robust predictions with uncertainty quantification.

### Business Impact
- **Marketing Budget Optimization**: Identify highest-ROI channels for acquisition spend
- **Player Segmentation**: Distinguish whales, dolphins, and minnows early in lifecycle
- **Churn Prediction**: Leverage survival analysis for retention modeling
- **Revenue Forecasting**: Project future revenue streams by cohort and channel

---

## 🚀 Quick Start

### Option 1: Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/ltv-forecasting.git
cd ltv-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run streamlit_app.py

# Or run the analysis pipeline
python run_pipeline.py
```

### Option 2: Deploy to Streamlit Cloud

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your forked repo
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

---

## 🧠 Models Implemented

### 1. BG/NBD Model (Beta-Geometric/Negative Binomial Distribution)
*Predicts transaction frequency and customer "alive" probability*

```
P(alive) = f(frequency, recency, tenure)
E[purchases] = λ × P(alive) × time_horizon
```

### 2. Gamma-Gamma Model
*Estimates expected average transaction value*

```
E[M|x, m̄, params] = (γ + x·m̄) / (q + x·p - 1)
```

### 3. Survival Analysis
- **Kaplan-Meier**: Non-parametric survival curves by channel
- **Cox Proportional Hazards**: Covariate effects on churn hazard

### 4. Gradient Boosting (Huber Loss)
*Robust to whale outliers, captures non-linear feature interactions*

### 5. Quantile Regression
*Uncertainty estimation for risk-aware budget allocation*

### 6. Ensemble Predictor
*Weighted combination optimizing for both accuracy and calibration*

```python
LTV_ensemble = 0.4 × LTV_probabilistic + 0.6 × LTV_gradient_boosting
```

---

## 📁 Project Structure

```
ltv_forecasting/
├── src/
│   └── ltv_models.py          # Core model implementations
├── outputs/
│   ├── dashboard.html         # Interactive executive dashboard
│   ├── dashboard_data.json    # Aggregated analytics data
│   ├── player_predictions.csv # Player-level predictions
│   ├── channel_insights.csv   # Channel performance metrics
│   └── feature_importance.csv # Feature rankings
├── run_pipeline.py            # Main execution script
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ltv-forecasting.git
cd ltv-forecasting

# Install dependencies
pip install -r requirements.txt

# Run the analysis
python run_pipeline.py
```

### Output
The pipeline generates:
- **Interactive Dashboard** (`outputs/dashboard.html`)
- **Player Predictions** with confidence intervals
- **Channel Performance Matrix**
- **Feature Importance Rankings**

---

## 🌐 Streamlit Deployment

### Local Development
```bash
streamlit run streamlit_app.py
```

### Deploy to Streamlit Cloud

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: LTV Forecasting Dashboard"
   git remote add origin https://github.com/yourusername/ltv-forecasting.git
   git push -u origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository: `yourusername/ltv-forecasting`
   - Branch: `main`
   - Main file: `streamlit_app.py`
   - Click "Deploy"

3. **Your app will be live at:**
   `https://ltv-forecasting.streamlit.app`

---

## 📈 Model Performance

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **R² Score** | 0.824 | 0.60 - 0.75 |
| **MAE** | $328 | $400 - $600 |
| **RMSE** | $1,359 | $1,500+ |
| **MAPE** | 20.3% | 25% - 35% |

---

## 🎯 Channel Insights

| Channel | Avg LTV | CAC | LTV:CAC | Performance |
|---------|---------|-----|---------|-------------|
| Referral | $712 | $5 | 142x | ⭐ Excellent |
| Organic Search | $516 | $0 | ∞ | ⭐ Excellent |
| Cross-Promo | $497 | $3 | 166x | ⭐ Excellent |
| Influencer | $453 | $25 | 18x | ✓ Good |
| Paid Social | $427 | $12.50 | 34x | ✓ Good |
| App Store | $340 | $8 | 42x | ✓ Good |

---

## 🔬 Methodology

### Feature Engineering

**RFM Features:**
- Recency (days since last purchase)
- Frequency (transaction count)
- Monetary (total spend, mean, std, max)

**Behavioral Signals:**
- Session frequency & duration
- Engagement consistency
- Social interaction ratio
- Achievement progression

**Early Indicators (Day 1-7):**
- First-week purchases
- Initial spend velocity
- Conversion timing

### Data Pipeline

```
Raw Transactions → Feature Engineering → Train/Val Split
                                              ↓
                        ┌─────────────────────┼─────────────────────┐
                        ↓                     ↓                     ↓
                    BG/NBD             Gamma-Gamma          Gradient Boosting
                        ↓                     ↓                     ↓
                        └──────────→ Ensemble Prediction ←──────────┘
                                              ↓
                              Channel Insights + Uncertainty Bounds
```

---

## 📊 Data Sources

This project simulates realistic gaming transaction data. For production use, connect to:

| Source | Description |
|--------|-------------|
| [Online Retail II - Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci) | Transaction history, Customer IDs |
| [Customer LTV Dataset](https://www.kaggle.com/datasets/iyadavvaibhav/customer-lifetime-value) | CAC, Channel, Revenue |
| [UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail) | Classic retail dataset |

### Gaming Reframe
| Traditional | Gaming Context |
|-------------|---------------|
| Purchases | In-game spend |
| Customers | Players |
| Channels | Paid social / organic / referral |

---

## 🛠 Technical Stack

- **Core**: Python 3.8+, NumPy, Pandas, SciPy
- **ML**: scikit-learn, Gradient Boosting
- **Visualization**: Chart.js, HTML5
- **Statistical**: Custom BG/NBD, Gamma-Gamma implementations

---

## 📝 Resume Signal

> **Developed LTV prediction models linking acquisition channels and early behavior to long-term player value.**

### Key Achievements:
- Implemented 5+ advanced modeling techniques (BG/NBD, Gamma-Gamma, Cox PH, Gradient Boosting, Quantile Regression)
- Built ensemble predictor achieving R² = 0.824 (outperforming industry benchmarks)
- Created executive dashboard for marketing budget optimization
- Engineered 20+ features including early behavioral indicators

### Skills Demonstrated:
- Statistical modeling (survival analysis, Bayesian methods)
- Machine learning (gradient boosting, ensemble methods)
- Feature engineering (RFM, behavioral signals)
- Data visualization (executive dashboards)
- Production code quality (OOP, type hints, documentation)

---

## 📄 License

MIT License - feel free to use this for your own portfolio!

---

## 🤝 Contributing

Contributions welcome! Areas for extension:
- [ ] Deep learning sequence models (LSTM/Transformer)
- [ ] Real-time prediction API
- [ ] A/B test integration
- [ ] Multi-touch attribution modeling

---

*Built with ❤️ for data science portfolios*
