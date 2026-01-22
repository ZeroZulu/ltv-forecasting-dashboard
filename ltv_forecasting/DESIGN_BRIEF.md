# LTV Analytics Dashboard - Design Brief

## Project Overview
**Project Name:** LTV Forecasting by Marketing Channel  
**Type:** Executive-Level Marketing Analytics Dashboard  
**Platform:** Web Application (Streamlit + React)

---

## Design Specifications

### Style Direction
- **Primary Style:** Skeuomorphic with modern glass elements
- **Secondary Style:** Glassmorphism with deep shadows
- **Overall Feel:** Premium, data-rich, executive-level

### Color Palette

| Element | Color | Hex Code |
|---------|-------|----------|
| Background Primary | Deep Black | `#0a0a0c` |
| Background Secondary | Dark Gray | `#111113` |
| Card Background | Translucent Dark | `rgba(28, 28, 32, 0.9)` |
| Primary Accent | Vibrant Green | `#22c55e` |
| Secondary Accent | Light Green | `#4ade80` |
| Accent Glow | Green Shadow | `rgba(34, 197, 94, 0.3)` |
| Text Primary | Pure White | `#ffffff` |
| Text Secondary | Soft White | `rgba(255, 255, 255, 0.7)` |
| Text Muted | Dim White | `rgba(255, 255, 255, 0.4)` |
| Border Default | Subtle White | `rgba(255, 255, 255, 0.05)` |
| Border Accent | Green Tint | `rgba(34, 197, 94, 0.2)` |
| Status Active | Green | `#22c55e` |
| Status Warning | Amber | `#f59e0b` |
| Status Error | Red | `#ef4444` |

### Typography

| Element | Font | Weight | Size |
|---------|------|--------|------|
| Logo | Inter | 700 | 18px |
| Page Title | Inter | 700 | 28px |
| Card Title | Inter | 600 | 16px |
| KPI Value | JetBrains Mono | 700 | 32px |
| Body Text | Inter | 400 | 14px |
| Labels | Inter | 600 | 11px |
| Badges | Inter | 500 | 12px |

---

## Layout Structure

### 3-Column Grid System
```
┌─────────────────────────────────────────────────────────┐
│  Sidebar (280px)  │         Main Content Area           │
│                   │                                     │
│  ┌─────────────┐  │  ┌───────────────────────────────┐  │
│  │    Logo     │  │  │          Header Bar           │  │
│  ├─────────────┤  │  ├───────────────────────────────┤  │
│  │   Search    │  │  │     KPI Cards (3 columns)     │  │
│  ├─────────────┤  │  ├──────────────┬────────────────┤  │
│  │             │  │  │   Channel    │   LTV Trend    │  │
│  │  Navigation │  │  │   ROI Card   │   Chart Card   │  │
│  │    Menu     │  │  │              │                │  │
│  │             │  │  ├──────────────┴────────────────┤  │
│  ├─────────────┤  │  │                               │  │
│  │   Features  │  │  │      Player Data Table        │  │
│  │             │  │  │                               │  │
│  ├─────────────┤  │  │                               │  │
│  │    Tools    │  │  └───────────────────────────────┘  │
│  ├─────────────┤  │                                     │
│  │   Upgrade   │  │                                     │
│  │    Card     │  │                                     │
│  └─────────────┘  │                                     │
└─────────────────────────────────────────────────────────┘
```

---

## Component Specifications

### 1. Glass Card Component
```css
.glass-card {
  background: linear-gradient(145deg, rgba(28, 28, 32, 0.9), rgba(18, 18, 22, 0.95));
  backdrop-filter: blur(20px);
  border-radius: 20px;
  border: 1px solid rgba(255, 255, 255, 0.05);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4), 
              inset 0 1px 0 rgba(255, 255, 255, 0.05);
}
```

### 2. Skeuomorphic Button (Primary)
```css
.btn-primary {
  background: linear-gradient(145deg, #22c55e, #16a34a);
  color: #000;
  border-radius: 12px;
  box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4),
              inset 0 1px 0 rgba(255, 255, 255, 0.2);
}
```

### 3. Navigation Item (Active State)
```css
.nav-item.active {
  background: linear-gradient(145deg, 
    rgba(34, 197, 94, 0.15), 
    rgba(34, 197, 94, 0.05));
  border: 1px solid rgba(34, 197, 94, 0.2);
}
```

### 4. KPI Card Structure
- Header: Label + Icon
- Value: Large monospace number
- Footer: Trend badge + Comparison text

### 5. Data Table
- Rounded corners on container
- Hover state on rows
- Status badges with dot indicators
- Channel icons in cells

---

## Interaction Patterns

### Hover States
- Cards: Subtle lift (translateY: -2px)
- Buttons: Increased shadow + slight scale
- Table rows: Background highlight
- Navigation: Background tint

### Transitions
- All transitions: 0.2s ease
- Chart animations: 1s ease-out
- Progress bars: 1s ease

### Active States
- Green accent color
- Increased shadow glow
- Bold text weight

---

## Data Visualization Guidelines

### Charts
- **Line Charts:** Gradient fill below line, smooth curves
- **Bar Charts:** Rounded corners (6px), green fill
- **Area Charts:** Gradient from accent to transparent

### Progress Bars
- Track: `rgba(255, 255, 255, 0.05)`
- Fill: Green gradient with glow shadow
- Height: 8px
- Border radius: 4px

### Status Indicators
| Status | Color | Background |
|--------|-------|------------|
| Active | `#22c55e` | `rgba(34, 197, 94, 0.15)` |
| At Risk | `#f59e0b` | `rgba(245, 158, 11, 0.15)` |
| Churned | `#ef4444` | `rgba(239, 68, 68, 0.15)` |

---

## Responsive Breakpoints

| Breakpoint | Layout Changes |
|------------|----------------|
| > 1200px | Full 3-column layout |
| 768-1200px | Stack middle cards vertically |
| < 768px | Hide sidebar, single column |

---

## Accessibility Considerations

- Minimum contrast ratio: 4.5:1 for text
- Focus indicators on all interactive elements
- Keyboard navigation support
- Screen reader labels on icons

---

## File Structure for Implementation

```
ltv-forecasting/
├── components/
│   ├── GlassCard.tsx
│   ├── SkeuButton.tsx
│   ├── NavItem.tsx
│   ├── KPICard.tsx
│   ├── StatusBadge.tsx
│   └── ChannelBar.tsx
├── styles/
│   └── globals.css
├── pages/
│   └── dashboard.tsx
└── data/
    └── dashboard_data.json
```

---

## Reference Implementation

The design is based on the Qiespend dashboard template with adaptations for:
- Gaming/player-centric terminology
- LTV-specific metrics and KPIs
- Marketing channel performance visualization
- Predictive model accuracy indicators

---

## Deliverables

1. ✅ React Dashboard Component (`LTVDashboard.jsx`)
2. ✅ Standalone HTML Dashboard (`ltv_dashboard_skeuomorphic.html`)
3. ✅ Streamlit Application (`streamlit_app.py`)
4. ✅ Design Brief Documentation (this file)

---

*Design Brief v1.0 | LTV Analytics Platform*
