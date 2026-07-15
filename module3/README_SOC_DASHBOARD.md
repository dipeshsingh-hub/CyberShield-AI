# CyberShield SOC Dashboard — Professional Redesign

## Overview

The CyberShield Module 3 dashboard has been completely redesigned as a **professional Security Operations Center (SOC) interface** with enterprise-grade UX/UI, modern glassmorphism design, and SIEM-style threat monitoring capabilities.

## Design Principles

### 1. **Dark Theme with Professional Accents**
- **Primary Background**: `#0D1117` (Dark Navy)
- **Secondary Background**: `#161B22` (Slightly Lighter)
- **Primary Accent**: `#00D9FF` (Cyan) — for alerts, highlights, interactions
- **Secondary Accent**: `#0969DA` (Blue) — for secondary highlights
- **Danger Color**: `#DA3633` (Red) — for critical threats
- **Text**: `#E6EDF3` (Light gray) — optimized for readability on dark backgrounds

### 2. **Glassmorphism Cards**
- Background: `rgba(22, 27, 34, 0.7)` with `backdrop-filter: blur(10px)`
- Border: `1px solid rgba(48, 54, 61, 0.5)`
- Border Radius: `12px` for modern, rounded appearance
- Smooth transitions on hover:
  - Background opacity increases
  - Border color shifts to cyan
  - Subtle lift effect (`translateY(-2px)`)
  - Enhanced shadow with cyan glow

### 3. **Responsive Layout**
- **Top Navigation**: Logo, Current Time, System Status, Threat Status, Model Status
- **KPI Row**: 6 animated metrics (Total Events, Threats Detected, Critical Alerts, Avg Risk Score, Blocked Attacks, System Health)
- **Analysis Section**: 2-column layout (Threat Timeline + Risk Distribution)
- **Heatmap**: Attack frequency patterns across time and categories
- **Alerts Table**: Interactive, searchable, filterable, color-coded by severity

## Component Breakdown

### Top Navigation Bar
```
[🛡️ CyberShield SOC] | [Current Time] | [System Status] | [Threat Level] | [Model Status]
```
- Fixed height, glassmorphic background
- Real-time updates
- Status indicators with pulse animation
- Color-coded health/threat levels

### KPI Cards
Each card displays:
- **Icon** + **Label** (uppercase, letter-spaced)
- **Large Value** (32px, cyan color)
- **Trend Arrow** (↑ green for positive, ↓ red for negative)
- **Mini Sparkline** (implied through trend percentage)
- **Colored Border** (cyan on hover)
- **Hover Animation**: Scale 1.02, shadow glow

KPIs:
1. **Total Events**: Count of all processed security events
2. **Threats Detected**: Count of anomalies flagged by ML models
3. **Critical Alerts**: Events classified as "Critical" risk
4. **Average Risk Score**: Mean of all final_risk_probability values
5. **Blocked Attacks**: Confirmed attack events
6. **System Health**: Overall system operational status (%)

### Middle Section

#### Left: Interactive Threat Timeline (Plotly)
- **X-axis**: Timestamp
- **Y-axis**: Risk Score (0-100%)
- **Markers**: Individual events, color-coded by risk level
  - 🟢 Low (green)
  - 🟡 Medium (orange)
  - 🔴 Critical (red)
- **Overlay Line**: 20-event rolling average (cyan dashed)
- **Hover**: Shows event details, timestamp, risk category
- **Interactive**: Zoom, pan, hover tooltips

#### Right: Risk Distribution Donut Chart
- **Segments**: Low, Medium, Critical
- **Colors**: Matching risk colors
- **Inner Ring**: 40% to show percentage clearly
- **Labels**: Risk level + percentage
- **Hover**: Count and percentage details

### Attack Heatmap
- **Rows**: Attack categories (Network, Email, API, Web, Insider, Malware)
- **Columns**: Days of the week
- **Color Scale**: Green (low) → Orange (medium) → Red (high)
- **Purpose**: Identify patterns in attack types by time
- **Interactive**: Hover for exact counts

### Recent Alerts Table
- **Features**:
  - **Search**: By IP address or channel
  - **Filter**: By risk level (Low/Medium/Critical)
  - **Filter**: By channel (email, network, api, etc.)
  - **Sort**: Click column headers
  - **Color Coding**: Rows highlighted by severity
    - Critical: Red tint + left border
    - High: Orange tint + left border
    - Medium: Blue tint + left border
    - Low: Green tint + left border

- **Columns**:
  - Timestamp: When the event occurred
  - Source IP: Originating host
  - Dest IP: Target host
  - Channel: Type (email, network, api, etc.)
  - Risk: Category with emoji indicator
  - Score %: Numerical risk percentage (0-100)
  - Attack: Boolean (Yes/No) confirmed attack

## Styling Features

### Animations
1. **Pulse Animation**: Status indicators gently pulse (2s cycle)
2. **Hover Lift**: Cards lift on hover (2px upward)
3. **Scale Transform**: KPI cards scale 1.02x on hover
4. **Glow Effect**: Cyan shadow extends on card hover
5. **Color Transition**: Border color smoothly transitions to cyan

### Spacing
- **Card Padding**: 20-24px
- **Dividers**: 1px border with reduced opacity, 32px vertical margin
- **Component Gaps**: 16px between cards
- **Section Padding**: 40px at top for separation

### Typography
- **Title**: 28px, bold, cyan color, flex layout
- **Subheadings**: 20px, border-bottom accent
- **KPI Labels**: 12px, uppercase, letter-spaced
- **KPI Values**: 32px, bold, cyan
- **Body Text**: 14px, light gray
- **Small Text**: 11-12px for metadata, reduced opacity

## Plotly Configurations

### All Charts Share:
```python
template="plotly_dark"
plot_bgcolor="rgba(22, 27, 34, 0.5)"
paper_bgcolor="rgba(22, 27, 34, 0.3)"
```

### Grid Lines
- Color: `rgba(48, 54, 61, 0.2)` (subtle gray)
- Width: 1px
- Provides visual reference without clutter

### Legends
- Background: Semi-transparent dark (`rgba(22, 27, 34, 0.7)`)
- Border: Cyan with 0.3 opacity
- Positioned at top-left (or bottom for tables)

### Hover Behavior
- Template: `"x unified"` for timeline (all series on one line)
- Shows: Value, timestamp, category
- Background: Dark with high contrast text

## Data Transformations

### Threat Timeline
```python
timeline_df["rolling_risk"] = timeline_df["final_risk_probability"].rolling(20, min_periods=1).mean()
```
Smooths individual data points to reveal trends.

### Attack Heatmap
```python
df_heatmap["hour"] = pd.to_datetime(df_heatmap["timestamp"]).dt.hour
df_heatmap["day"] = pd.to_datetime(df_heatmap["timestamp"]).dt.day_name()
df_heatmap["attack_category"] = np.random.choice(attack_categories, ...)
```
Randomly assigns attack categories (in production, derive from actual threat intel).

### Alerts Table
- Sorted by timestamp (descending)
- Filtered by risk level and channel
- Searchable by IP or channel name
- Top 50 most recent alerts

## Performance Optimizations

### Caching
- Data loaded once with `@st.cache_data(ttl=300)` (5-minute TTL)
- Charts regenerated only on data refresh
- Layout elements render instantly

### Chart Rendering
- Plotly renders client-side (no server strain)
- All CSS in single `<style>` block (minimal DOM)
- Responsive grid layout (no JavaScript frameworks)

## Browser Compatibility

- ✅ Chrome/Edge (latest)
- ✅ Firefox (latest)
- ✅ Safari 14+
- ✅ Responsive on tablets (768px+)
- ✅ Dark theme detection (respects OS preference)

## Accessibility

- Color contrast ratios meet WCAG AA standards
- Icons paired with text labels
- Hover states clearly indicate interactivity
- Table rows have color + border (not color alone)
- Keyboard navigation supported in Plotly charts

## Customization

### Change Accent Colors
Edit the color constants at the top of `app.py`:
```python
COLOR_PRIMARY = "#00D9FF"  # Cyan
COLOR_ACCENT = "#0969DA"  # Blue
COLOR_DANGER = "#DA3633"  # Red
```

### Adjust Card Styling
Modify the `.soc-card` and `.kpi-card` CSS classes:
```css
.soc-card {
    background: rgba(22, 27, 34, 0.7);  /* opacity */
    backdrop-filter: blur(10px);        /* blur amount */
    border-radius: 12px;                /* corner roundness */
    padding: 24px;                      /* internal spacing */
}
```

### Extend Data Transformations
Add additional metrics in the data loading section:
```python
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
df["severity"] = pd.cut(df["final_risk_probability"], ...)
```

## Running the Dashboard

```bash
cd module3/src
streamlit run app.py --theme.base dark --logger.level=error
```

### Environment
- Python 3.8+
- Streamlit 1.28+
- Plotly 5.0+
- Pandas 1.5+
- NumPy 1.23+

## Future Enhancements

1. **Real-Time Updates**: WebSocket connection for live threat feeds
2. **Drill-Down**: Click events to see full details in modal
3. **Export**: Download alerts as CSV/PDF
4. **Custom Dashboards**: User-configurable widget layouts
5. **Alerts**: Email/Slack notifications on critical events
6. **Incident Correlation**: Show related alerts and patterns
7. **Threat Intelligence**: Integrate external feeds (VirusTotal, etc.)
8. **Playbook Automation**: Trigger response workflows from dashboard

## Troubleshooting

### Charts Not Rendering
- Clear browser cache
- Restart Streamlit: `streamlit cache clear`
- Check console for JavaScript errors

### Data Not Updating
- Verify `unified_threat_data.csv` exists
- Run `build_dataset.py` first
- Check file permissions

### Performance Issues
- Reduce TTL on `@st.cache_data` if data changes frequently
- Filter table to fewer rows
- Use `--logger.level=error` to reduce overhead

## Credits

Designed with inspiration from:
- Datadog SOC dashboard
- Splunk Enterprise Security
- Elastic Security
- Modern glassmorphism design trends

---

**Last Updated**: 2026-07-15
