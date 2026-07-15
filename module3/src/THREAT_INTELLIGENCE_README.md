# CyberShield AI - Threat Intelligence Dashboard

A comprehensive threat intelligence dashboard for the CyberShield SOC platform with real-time CVE tracking, malware analysis, threat actor monitoring, and global attack visualization.

## 🎯 Features

### 1. **Top CVEs Dashboard**
- **CVE Listing**: Display of critical vulnerabilities with:
  - CVE ID and title
  - CVSS scores with color-coded severity
  - Publication dates
  - Known exploit counts
  - Affected systems count
- **Visualizations**:
  - CVSS score distribution (horizontal bar chart)
  - Affected systems by CVE (vertical bar chart)
  - Severity breakdown
- **Expandable Cards**: Click to view full CVE descriptions
- **Real-time Updates**: Data refreshed hourly

### 2. **MITRE ATT&CK Framework**
- **Tactic Coverage**: All 14 MITRE ATT&CK tactics:
  - Reconnaissance
  - Resource Development
  - Initial Access
  - Execution
  - Persistence
  - Privilege Escalation
  - Defense Evasion
  - Credential Access
  - Discovery
  - Lateral Movement
  - Collection
  - Command & Control
  - Exfiltration
  - Impact
- **Visualizations**:
  - Technique count by tactic (bar chart)
  - Observed technique sunburst diagram
  - Tactic activity heatmap
- **Interactive Matrix**: Drill-down into specific tactics

### 3. **Threat Actor Intelligence**
- **Actor Profiles**: Detailed information on tracked APT groups:
  - Actor name and aliases
  - Country of origin
  - Active since (years)
  - Target sectors
  - MITRE techniques used
  - Last observed activity
  - Attribution confidence
- **Notable Actors**:
  - APT-28 (Fancy Bear) - Russia
  - APT-29 (Cozy Bear) - Russia
  - APT-41 (Winnti) - China
  - Lazarus Group - North Korea
- **Visualizations**:
  - Techniques by actor (horizontal bar)
  - Actor timeline with activity levels
  - Confidence score distribution

### 4. **Malware Family Tracking**
- **Malware Families**: Comprehensive malware intelligence:
  - Family name and classification
  - Malware type (Trojan, Ransomware, Botnet, etc.)
  - First discovery date
  - Variant count
  - Known infections
  - Last detected date
  - Threat level
  - Attribution (if known)
- **Notable Malware**:
  - Emotet (Banking Trojan)
  - Mirai (Botnet)
  - Wannacry (Ransomware)
  - Trickbot (Banking Trojan)
- **Visualizations**:
  - Infection distribution (pie chart)
  - Variants by family (bar chart)
  - Family timeline

### 5. **Indicators of Compromise (IoCs)**
- **IoC Types**: Multiple indicator categories:
  - IP Addresses
  - Domains
  - URLs
  - File Hashes (MD5, SHA-1, SHA-256)
  - Email addresses
  - Registry keys
  - File paths
- **IoC Intelligence**:
  - First and last seen timestamps
  - Associated malware families
  - Associated threat actors
  - Detection count across sensors
  - Severity level
- **Visualizations**:
  - IoC type distribution (pie chart)
  - Detection activity (bar chart)
  - Timeline of detections
- **Expandable Cards**: Full IoC details and timeline

### 6. **Interactive World Map**
- **Attack Origins**: Global attack visualization with:
  - Geolocation of source IPs
  - Attack frequency by country
  - Bubble size = attack volume
  - Color gradient (green to red) = intensity
- **Supported Regions**: 8+ major attack-originating countries
  - Russia
  - China
  - Iran
  - North Korea
  - India
  - Brazil
  - Romania
  - Vietnam
- **Interactive Features**:
  - Hover for country details
  - Click to filter by region
  - Zoom and pan capabilities

### 7. **Country Attack Heatmap**
- **Attack Distribution**: Bar chart showing:
  - Top attacking countries
  - Attack frequency per country
  - Color-coded by attack volume
  - Sortable by count
- **Percentage Breakdown**: Pie chart of attack origins
- **Trend Analysis**: Attack patterns over time

### 8. **Risk Evolution Timeline**
- **Risk Score Graph**: 
  - 45-85% dynamic risk scoring
  - Fill area for visual impact
  - Trend line overlay
  - Hover tooltips with exact values
- **Cumulative Threat Count**:
  - Growing threat landscape
  - Cumulative indicator of total threats discovered
- **Incident Activity**:
  - Daily incident count
  - Bar chart visualization
  - Correlation with risk score
- **Time Range**: 45-day historical view (configurable)

## 🎨 Design System

### Colors
```python
COLOR_PRIMARY = "#00D9FF"      # Cyan - Primary highlights
COLOR_ACCENT = "#0969DA"      # Blue - Secondary elements
COLOR_DANGER = "#DA3633"      # Red - Critical/danger
COLOR_WARNING = "#D29922"     # Orange - Warnings
COLOR_SUCCESS = "#1a7f37"     # Green - Success
COLOR_DARK_BG = "#0D1117"     # Dark background
COLOR_CARD_BG = "#161B22"     # Card background
```

### Severity Levels
```
Critical: #DA3633 (Red)
High:     #D29922 (Orange)
Medium:   #0969DA (Blue)
Low:      #1a7f37 (Green)
```

### Professional SOC Styling
- **Glassmorphism Cards**: Blur effect with transparency
- **Smooth Transitions**: 0.3s ease animations
- **Hover Effects**: 
  - Background opacity increase
  - Border color highlight (cyan)
  - Subtle transform lift (translateY -2px)
  - Enhanced shadow glow
- **Responsive Layout**: Works on tablets and mobile

## 📊 Visualizations (All Plotly)

### Chart Types Used
1. **Bar Charts**: CVSS scores, affected systems, techniques, variants
2. **Pie Charts**: Malware distribution, IoC types, attack origins
3. **Sunburst**: MITRE tactic hierarchy
4. **Scatter**: Actor timeline, confidence vs techniques
5. **Geo Map**: Global attack origins with markers
6. **Area Charts**: Risk evolution, cumulative threats
7. **Subplots**: Multi-axis visualization (risk + incidents)
8. **Heatmaps**: Technique coverage matrix

### Interactive Features
- Hover tooltips on all charts
- Zoom and pan enabled
- Click to filter/drill-down
- Color scales for correlation
- Legend toggling

## 🔄 Data Sources

### Internal Sources
- **Module 1**: Network anomaly detections
- **Module 2**: Phishing and threat scores
- **Dashboard Cache**: Historical threat data

### External Sources (Simulated)
- **NVD**: CVE database
- **MITRE**: ATT&CK framework
- **Threat Intelligence Feeds**:
  - VirusTotal
  - AlienVault OTX
  - Shodan
  - URLhaus
  - PhishTank

### Real-time Updates
- CVE updates every 6 hours
- Malware signatures updated hourly
- IoCs refreshed every 30 minutes
- Actor activity monitored continuously

## 🔧 Configuration

### Streamlit Settings
```bash
streamlıt run threat_intelligence.py --theme.base dark --logger.level=error
```

### Data Refresh Intervals
```python
@st.cache_data(ttl=3600)  # 1 hour cache for CVEs
@st.cache_data(ttl=1800)  # 30 min cache for IoCs
@st.cache_data(ttl=600)   # 10 min cache for risk scores
```

### Customization

#### Add Custom CVEs
Edit `generate_cve_data()` function:
```python
def generate_cve_data():
    cves = [
        {
            "CVE ID": "CVE-2024-XXXX",
            "Title": "Vulnerability Title",
            "CVSS Score": 9.5,
            "Severity": "Critical",
            # ...
        },
    ]
    return pd.DataFrame(cves)
```

#### Modify Color Scheme
Edit color constants at top:
```python
COLOR_PRIMARY = "#00D9FF"  # Change primary color
COLOR_DANGER = "#DA3633"  # Change danger color
```

#### Adjust Cache Times
```python
@st.cache_data(ttl=7200)  # 2 hours instead of 1
```

## 📈 Key Metrics

### CVE Metrics
- CVSS Score (0-10)
- Exploit count
- Affected systems
- Publication date
- Time to patch

### Actor Metrics
- Active techniques
- Confidence score (0-100%)
- Years active
- Targets per actor
- Last activity

### Malware Metrics
- Variant count
- Known infections
- Detection rate
- Propagation method
- Evasion techniques

### IoC Metrics
- Detection count
- Geographic spread
- Time to detection
- Associated threats
- False positive rate

### Risk Metrics
- Overall risk score (0-100%)
- Threat velocity (trends)
- Attack intensity
- Incident rate
- Mean time to detection (MTTD)

## 🚀 Deployment

### Local Development
```bash
cd module3/src
streamlit run threat_intelligence.py
```

### Docker
```dockerfile
FROM python:3.10
RUN pip install streamlit plotly pandas numpy
COPY threat_intelligence.py .
CMD ["streamlit", "run", "threat_intelligence.py"]
```

### Cloud Deployment
**Streamlit Cloud**: Connect GitHub repo, select this file
**AWS/Azure**: Deploy Docker container
**On-premise**: Run with Docker or direct Python

## 🔐 Security Considerations

- ✅ All data sanitized before display
- ✅ No sensitive credentials in code
- ✅ HTTPS enforced for all links
- ✅ Cache isolation per user (if deployed multi-user)
- ✅ Read-only access to threat data
- ✅ Audit logging for sensitive queries

## 📱 Responsive Design

- **Desktop**: Full layout with all visualizations
- **Tablet**: 2-column grid, stacked on small tablets
- **Mobile**: Single column, vertical stack
- **Minimum Width**: 400px (optimized for iPhone SE)

## 🔍 Search & Filter Capabilities

### Planned Features (v2.0)
- Full-text search across all threat data
- Advanced filtering by:
  - Date range
  - Severity level
  - Actor/malware/IoC association
  - Geographic origin
  - Impact scope
- Saved searches and alerts
- Custom reports

## 📊 Analytics & Reporting

### Export Formats
- CSV (all tables)
- PDF (formatted reports)
- JSON (API consumption)
- Excel (multi-sheet workbooks)

### Report Templates
- Executive summary
- Detailed threat analysis
- IoC watchlist
- APT trend analysis
- Risk assessment

## 🐛 Troubleshooting

### Charts Not Rendering
- Ensure Plotly is updated: `pip install --upgrade plotly`
- Check browser console for errors
- Clear cache: `streamlit cache clear`

### Data Not Updating
- Check cache TTL values
- Verify data source connectivity
- Review logs for import errors

### Performance Issues
- Reduce time range on timeline
- Limit number of IoCs displayed
- Increase cache TTL values
- Profile with Streamlit's built-in profiler

## 📚 Additional Resources

- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [NVD - National Vulnerability Database](https://nvd.nist.gov/)
- [CVSS Calculator](https://www.first.org/cvss/calculator/3.1)
- [Plotly Documentation](https://plotly.com/python/)

## 🤝 Integration Points

### With CyberShield Modules
- **Module 1**: Network anomalies → Attack origins
- **Module 2**: Phishing threats → Malware association
- **Module 3**: Dashboard → Threat visualization
- **Module 4**: Orchestrator → IoC-triggered playbooks

### External APIs (for future)
- VirusTotal API for file hashes
- Shodan for IoT device tracking
- AlienVault OTX for threat feeds
- MITRE ATT&CK API for techniques

## 📞 Support

For issues or feature requests:
- GitHub Issues: https://github.com/gunjit007/CyberShield-AI/issues
- Discussions: https://github.com/gunjit007/CyberShield-AI/discussions

---

**Last Updated**: 2026-07-15

🔴 **CyberShield AI** - Threat Intelligence Dashboard
