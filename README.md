# 🎮 LTV Forecasting Dashboard

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=flat&logo=plotly&logoColor=white)](https://plotly.com)
[![Tableau](https://img.shields.io/badge/Tableau-Dashboard-E97627?style=flat&logo=tableau&logoColor=white)](#tableau-dashboard)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Predict player lifetime value by marketing channel using machine learning.** Built for gaming analytics teams to optimize user acquisition spend and identify high-value player segments.

![Dashboard Preview](https://img.shields.io/badge/Status-Live-brightgreen) ![Model R²](https://img.shields.io/badge/Model_R²-0.891-blue)

---

## 📊 Key Results

| Metric | Value | Benchmark |
|--------|-------|-----------|
| **R² Score** | 0.891 | 0.60-0.75 |
| **Cross-Val R²** | 0.891 ± 0.057 | — |
| **MAE** | $53.23 | — |
| **Top Predictor** | Early 7-day spend (89.8%) | — |

**Business Insight:** Whales (5% of players) generate 85% of revenue. Referral channel delivers highest LTV ($714) with 143x ROI.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/ZeroZulu/ltv-forecasting-dashboard.git
cd ltv-forecasting-dashboard/ltv_forecasting

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run streamlit_app.py
```

---

## 🏗️ Project Structure

```
ltv_forecasting/
├── notebooks/
│   └── LTV_Analysis_v3_FIXED.ipynb   # Main analysis notebook
├── src/
│   └── ltv_models.py                  # Model classes
├── data/                              # Generated datasets
├── outputs/                           # Model artifacts
├── streamlit_app.py                   # Interactive dashboard
└── requirements.txt
```

---

## 📈 Dashboards

### Streamlit Dashboard (Live)

Interactive web app with real-time filtering by time period and channel.

**Features:**
- 📊 KPI cards (Revenue, LTV, R², Players)
- 📈 LTV trend visualization
- 🎯 Channel performance matrix
- 👥 Player segment analysis
- 🤖 Model performance metrics

```bash
streamlit run streamlit_app.py
```

### Tableau Dashboard

> 🚧 **Coming Soon** — Executive-level visualizations for stakeholder presentations.

<!-- TODO: Add Tableau Public link -->
<!-- [View on Tableau Public](https://public.tableau.com/app/profile/YOUR_PROFILE/viz/LTV_Dashboard) -->

---

## 🔬 Methodology

### Data
- **10,000 players** across 6 acquisition channels
- **35,000+ transactions** with realistic spending patterns
- Segments: Whale (2%), Dolphin (8%), Minnow (30%), F2P (60%)

### Model
- **Algorithm:** Gradient Boosting Regressor (scikit-learn)
- **Features:** Early behavior signals (first 7 days), tenure, channel
- **Validation:** 5-fold cross-validation

### Key Improvements (v3)
- ✅ Removed feature leakage (monetary features)
- ✅ Added regularization to prevent overfitting
- ✅ Uses only predictive signals available at acquisition

---

## 📡 Channel Performance

| Channel | Avg LTV | CAC | ROI | Recommendation |
|---------|---------|-----|-----|----------------|
| Referral | $714 | $5 | 143x | 🟢 Scale |
| Organic | $518 | $0 | ∞ | 🟢 Scale |
| Cross Promo | $499 | $3 | 166x | 🟢 Scale |
| Influencer | $455 | $25 | 18x | 🟡 Optimize |
| Paid Social | $429 | $12.50 | 34x | 🔵 Maintain |
| App Store | $341 | $8 | 43x | 🔵 Maintain |

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **Visualization:** Plotly, Matplotlib, Seaborn
- **Dashboard:** Streamlit
- **BI:** Tableau (coming soon)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <b>Built for data-driven UA optimization</b><br>
  <a href="#-quick-start">Get Started</a> •
  <a href="#-dashboards">View Dashboards</a> •
  <a href="#-methodology">Methodology</a>
</p>
