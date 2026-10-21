"""
Design System and Custom Styling for SpamAdapt Dashboard.

Aesthetics:
- Dark navy palette: #0a0f1d (deep background), #1e293b (cards), #334155 (borders)
- Accents: Electric Blue (#3b82f6), Neon Violet (#8b5cf6), Sky Blue (#38bdf8)
- Indicators: Emerald Green (#10b981) for Legitimate (HAM), Crimson Red (#ef4444) for Suspicious (SPAM)
- Clean, modern typography and responsive container spacing
"""

DARK_THEME_CSS = """
<style>
/* Global resets and background */
.stApp {
    background-color: #090d16;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}

/* Header container */
.main-header {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.5);
}

.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -0.02em;
}

.sub-title {
    color: #94a3b8;
    font-size: 1.0rem;
    margin-top: 6px;
    margin-bottom: 0px;
}

/* Model badge */
.badge-container {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.tech-badge {
    display: inline-flex;
    align-items: center;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    background-color: #1e293b;
    color: #cbd5e1;
    border: 1px solid #334155;
}

.tech-badge-primary {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.tech-badge-purple {
    background-color: rgba(139, 92, 246, 0.15);
    color: #c084fc;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

/* Dashboard Cards */
.dash-card {
    background: #131d31;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.card-title {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-bottom: 8px;
}

/* Prediction Result Banner */
.result-banner-ham {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 15px -1px rgba(16, 185, 129, 0.15);
}

.result-banner-spam {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 15px -1px rgba(239, 68, 68, 0.15);
}

.result-banner-review {
    background: linear-gradient(135deg, rgba(180, 83, 9, 0.35) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    box-shadow: 0 4px 15px -1px rgba(245, 158, 11, 0.15);
}

.result-tag-ham {
    color: #34d399;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.result-tag-spam {
    color: #f87171;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}

.result-tag-review {
    color: #fbbf24;
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Notice box */
.security-notice {
    background-color: rgba(30, 41, 59, 0.5);
    border-left: 3px solid #64748b;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: #94a3b8;
    border-radius: 0 6px 6px 0;
    margin-top: 12px;
}

/* Stat KPI Card */
.stat-box {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 8px;
    padding: 14px 16px;
    text-align: center;
}

.stat-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
}

.stat-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-top: 2px;
}

/* Streamlit button customization */
div.stButton > button:first-child {
    background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
    color: #ffffff;
    font-weight: 600;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    transition: all 0.2s ease-in-out;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-size: 0.95rem;
    font-weight: 600;
    color: #94a3b8;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #60a5fa !important;
}
</style>
"""
