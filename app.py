import base64
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sqlalchemy import create_engine


# ============================================================
# ASSETS & PAGE CONFIG
# ============================================================

LOGO_PATH = Path("deka-logo.png")
ICON_PATH = Path("deka-icon.png")

if ICON_PATH.exists():
    PAGE_ICON = Image.open(ICON_PATH)
elif LOGO_PATH.exists():
    PAGE_ICON = Image.open(LOGO_PATH)
else:
    PAGE_ICON = "🟡"

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon=PAGE_ICON,
    layout="wide"
)


# ============================================================
# BRAND COLORS
# ============================================================

NAVY = "#0B1026"
BLUE = "#1E2A4A"
GOLD = "#F2A93B"
CREAM = "#FAF8F2"
CARD = "#FFFFFF"
GREY = "#747B8D"
LINE = "#E7E0D6"
TAUPE = "#C8BFB2"
GREEN = "#2E7D5B"
RED = "#D95F59"


# ============================================================
# IMAGE HELPERS
# ============================================================

def image_to_base64(path: Path):
    if not path.exists():
        return None
    return base64.b64encode(path.read_bytes()).decode()


logo_base64 = image_to_base64(LOGO_PATH)

if logo_base64:
    LOGO_HTML = f"""
    <img src="data:image/png;base64,{logo_base64}" class="brand-logo" />
    """
    SIDEBAR_LOGO_HTML = f"""
    <div class="sidebar-logo-wrap">
        <img src="data:image/png;base64,{logo_base64}" class="sidebar-logo" />
    </div>
    """
else:
    LOGO_HTML = """
    <div class="brand-fallback"><span>deka</span><br>insight</div>
    """
    SIDEBAR_LOGO_HTML = """
    <div class="sidebar-logo-fallback"><span>deka</span><br>insight</div>
    """


# ============================================================
# CSS
# ============================================================

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 8% 5%, rgba(242,169,59,0.10), transparent 24%),
                linear-gradient(180deg, #FFFCF7 0%, {CREAM} 100%);
            color: {NAVY};
        }}

        header[data-testid="stHeader"] {{
            background: rgba(255,255,255,0.72);
            backdrop-filter: blur(10px);
        }}

        .main .block-container {{
            max-width: 1360px;
            padding-top: 1.2rem;
            padding-bottom: 4rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #ECE7DD;
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {{
            color: {BLUE};
        }}

        .sidebar-logo-wrap {{
            width: 118px;
            margin: 18px 0 28px 0;
            overflow: visible;
        }}

        .sidebar-logo {{
            width: 118px;
            height: auto;
            display: block;
            object-fit: contain;
        }}

        .sidebar-logo-fallback {{
            color: {NAVY};
            font-size: 1.25rem;
            font-weight: 900;
            line-height: 0.9;
            margin-bottom: 28px;
        }}

        .sidebar-logo-fallback span {{
            color: {GOLD};
        }}

        .hero {{
            position: relative;
            background:
                radial-gradient(circle at top right, rgba(242,169,59,0.13), transparent 26%),
                linear-gradient(135deg, #FFFFFF 0%, #FFF8EC 100%);
            border: 1px solid #EFE1CB;
            border-radius: 30px;
            padding: 32px 40px 34px 40px;
            box-shadow: 0 22px 56px rgba(11,16,38,0.07);
            margin-bottom: 28px;
            overflow: hidden;
        }}

        .brand-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 18px;
            margin-bottom: 22px;
            position: relative;
            z-index: 2;
        }}

        .brand-left {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .brand-logo {{
            width: 104px;
            max-height: 46px;
            object-fit: contain;
            display: block;
        }}

        .brand-fallback {{
            font-weight: 900;
            color: {NAVY};
            line-height: 0.9;
        }}

        .brand-fallback span {{
            color: {GOLD};
        }}

        .brand-sub {{
            color: {GREY};
            font-size: 0.80rem;
            font-weight: 800;
            letter-spacing: 0.05em;
            text-transform: uppercase;
        }}

        .brand-chip {{
            background: {NAVY};
            color: #FFFFFF;
            padding: 9px 16px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 0.80rem;
            white-space: nowrap;
        }}

        .eyebrow {{
            color: {GOLD};
            font-size: 0.76rem;
            font-weight: 900;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            margin-bottom: 12px;
            position: relative;
            z-index: 2;
        }}

        .hero-title {{
            color: {NAVY};
            font-size: 2.85rem;
            line-height: 1.03;
            font-weight: 900;
            letter-spacing: -0.055em;
            margin-bottom: 12px;
            max-width: 930px;
            position: relative;
            z-index: 2;
        }}

        .hero-accent {{
            color: {GOLD};
            font-style: italic;
            font-weight: 800;
            letter-spacing: -0.04em;
        }}

        .hero-copy {{
            color: {BLUE};
            font-size: 1.00rem;
            line-height: 1.62;
            max-width: 820px;
            position: relative;
            z-index: 2;
        }}

        .mini-flow {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
            position: relative;
            z-index: 2;
        }}

        .mini-flow span {{
            background: rgba(255,255,255,0.76);
            border: 1px solid #EFE1CB;
            color: {BLUE};
            border-radius: 999px;
            padding: 8px 12px;
            font-size: 0.80rem;
            font-weight: 800;
        }}

        .mini-flow b {{
            color: {GOLD};
        }}

        .section-title {{
            color: {NAVY};
            font-size: 1.42rem;
            font-weight: 900;
            letter-spacing: -0.035em;
            margin: 4px 0 4px 0;
        }}

        .section-subtitle {{
            color: {GREY};
            font-size: 0.92rem;
            margin-bottom: 14px;
        }}

        .metric-card {{
            background: {CARD};
            border: 1px solid {LINE};
            border-radius: 20px;
            padding: 18px 18px;
            box-shadow: 0 12px 28px rgba(11,16,38,0.05);
            height: 126px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-label {{
            color: {GREY};
            font-size: 0.70rem;
            font-weight: 900;
            letter-spacing: 0.10em;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: {NAVY};
            font-size: 1.84rem;
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1;
        }}

        .metric-note {{
            color: {GREY};
            font-size: 0.76rem;
        }}

        .verdict-card {{
            background: {NAVY};
            color: white;
            border-radius: 24px;
            padding: 24px 26px;
            min-height: 180px;
            box-shadow: 0 18px 42px rgba(11,16,38,0.16);
        }}

        .verdict-kicker {{
            color: {GOLD};
            font-size: 0.70rem;
            font-weight: 900;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 12px;
        }}

        .verdict-main {{
            color: white;
            font-size: 1.44rem;
            font-weight: 900;
            letter-spacing: -0.035em;
            line-height: 1.16;
            margin-bottom: 10px;
        }}

        .verdict-text {{
            color: #E5E8F0;
            font-size: 0.90rem;
            line-height: 1.52;
        }}

        .tier-card {{
            background: white;
            border: 1px solid {LINE};
            border-radius: 22px;
            padding: 22px 22px;
            min-height: 180px;
            box-shadow: 0 12px 28px rgba(11,16,38,0.05);
        }}

        .tier-title {{
            color: {NAVY};
            font-size: 0.98rem;
            font-weight: 900;
            margin-bottom: 10px;
        }}

        .tier-value {{
            color: {GOLD};
            font-size: 1.94rem;
            font-weight: 900;
            letter-spacing: -0.045em;
            margin-bottom: 8px;
        }}

        .tier-copy {{
            color: {BLUE};
            font-size: 0.86rem;
            line-height: 1.48;
        }}

        .signal-card {{
            background: white;
            border: 1px solid {LINE};
            border-left: 6px solid {GOLD};
            border-radius: 22px;
            padding: 20px 22px;
            min-height: 128px;
            box-shadow: 0 12px 28px rgba(11,16,38,0.05);
        }}

        .signal-title {{
            color: {NAVY};
            font-size: 1.00rem;
            font-weight: 900;
            margin-bottom: 8px;
        }}

        .signal-copy {{
            color: {BLUE};
            font-size: 0.90rem;
            line-height: 1.48;
        }}

        .pill {{
            display: inline-block;
            background: rgba(242,169,59,0.14);
            color: {NAVY};
            border: 1px solid rgba(242,169,59,0.34);
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.72rem;
            font-weight: 900;
            margin-bottom: 9px;
        }}

        .confidence {{
            display: inline-block;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.76rem;
            font-weight: 900;
            margin-top: 8px;
        }}

        .confidence-strong {{
            background: rgba(46,125,91,0.14);
            color: {GREEN};
            border: 1px solid rgba(46,125,91,0.30);
        }}

        .confidence-reliable {{
            background: rgba(242,169,59,0.16);
            color: {NAVY};
            border: 1px solid rgba(242,169,59,0.36);
        }}

        .confidence-directional {{
            background: rgba(217,95,89,0.10);
            color: {RED};
            border: 1px solid rgba(217,95,89,0.24);
        }}

        .hint {{
            color: {GREY};
            font-size: 0.84rem;
            margin-top: 10px;
            line-height: 1.45;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid {LINE};
            box-shadow: 0 12px 28px rgba(11,16,38,0.04);
        }}

        .stDownloadButton > button {{
            background: {GOLD};
            color: {NAVY};
            border: none;
            border-radius: 999px;
            padding: 0.68rem 1.18rem;
            font-weight: 900;
        }}

        .stDownloadButton > button:hover {{
            background: #E99D2F;
            color: {NAVY};
            border: none;
        }}

        hr {{
            border: none;
            height: 1px;
            background: {LINE};
            margin: 28px 0;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE
# ============================================================

@st.cache_resource
def get_engine():
    user = st.secrets["postgres"]["user"]
    password = quote_plus(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    database = st.secrets["postgres"]["database"]

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"
    return create_engine(url)


@st.cache_data(ttl=600)
def load_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM norm_value_all", engine)


df = load_data()


# ============================================================
# CLEANING
# ============================================================

numeric_cols = [
    "scale", "mean_score", "tb_pct", "t2b_pct", "t3b_pct",
    "base", "min_score", "max_score", "std_score",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object" or str(df[col].dtype) == "string":
        df[col] = df[col].astype("string").str.strip()
        df[col] = df[col].replace(["None", "nan", "NaN", "", "NULL"], pd.NA)

if "gender" in df.columns:
    df["gender"] = df["gender"].replace({
        "Perempuan": "Female",
        "Wanita": "Female",
        "Female": "Female",
        "Laki-Laki": "Male",
        "Laki-laki": "Male",
        "Laki Laki": "Male",
        "Pria": "Male",
        "Male": "Male",
    })

grade_order = ["Top 25%", "Average 50%", "Bottom 25%"]


# ============================================================
# LABELS & SORTING
# ============================================================

SLICE_LABEL_MAP = {
    "Global": "Overall Market",
    "category": "Category",
    "sub_category": "Sub Category",
    "detail_product": "Detail Product",
    "gender": "Gender",
    "age_group": "Age Group",
    "actual_age": "Age",
    "ses": "SES",
    "occupation": "Occupation",
    "type_of_study": "Type of Study",
    "test_type": "Test Type",
    "methodology": "Methodology",
    "sub_method": "Sub Method",
    "num_of_product": "# Product",
    "sequence": "Sequence",
    "category | gender": "Category × Gender",
    "category | age_group": "Category × Age Group",
    "category | ses": "Category × SES",
    "category | occupation": "Category × Occupation",
    "category | methodology": "Category × Methodology",
    "category | test_type": "Category × Test Type",
    "category | sub_category": "Category × Sub Category",
    "category | detail_product": "Category × Detail Product",
    "sub_category | gender": "Sub Category × Gender",
    "detail_product | gender": "Detail Product × Gender",
    "methodology | test_type": "Methodology × Test Type",
    "type_of_study | methodology": "Type of Study × Methodology",
}

VALUE_ORDER_MAP = {
    "age_group": ["<13", "13-17", "18-24", "25-34", "35-44", "45-54", "55+"],
    "gender": ["Female", "Male"],
    "norm_grade": grade_order,
    "scale": [5, 7, 9],
}


def clean_label(text):
    text = str(text)
    return SLICE_LABEL_MAP.get(text, text.replace("_", " ").title())


def sort_dropdown_values(col, values):
    if col in VALUE_ORDER_MAP:
        order = VALUE_ORDER_MAP[col]
        return sorted(values, key=lambda x: order.index(x) if x in order else len(order))

    if col in ["actual_age", "num_of_product", "sequence"]:
        return sorted(
            values,
            key=lambda x: float(x) if str(x).replace(".", "", 1).isdigit() else 999999,
        )

    return sorted(values)


# ============================================================
# DUPLICATE COLLAPSE
# ============================================================

def weighted_avg(series, weights):
    series = pd.to_numeric(series, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    mask = series.notna() & weights.notna() & (weights > 0)

    if not mask.any():
        return np.nan

    return np.average(series[mask], weights=weights[mask])


def collapse_duplicate_norms(data):
    dimension_cols = [
        "slice_type",
        "category",
        "sub_category",
        "detail_product",
        "gender",
        "age_group",
        "actual_age",
        "ses",
        "occupation",
        "type_of_study",
        "test_type",
        "methodology",
        "sub_method",
        "num_of_product",
        "sequence",
        "parameter_name",
        "scale",
        "norm_grade",
    ]

    group_cols = [c for c in dimension_cols if c in data.columns]
    metric_cols = ["mean_score", "tb_pct", "t2b_pct", "t3b_pct", "std_score"]

    def summarize(group):
        weights = pd.to_numeric(group["base"], errors="coerce").fillna(0)
        out = {}

        for metric in metric_cols:
            if metric in group.columns:
                out[metric] = weighted_avg(group[metric], weights)

        out["base"] = weights.sum()

        if "min_score" in group.columns:
            out["min_score"] = pd.to_numeric(group["min_score"], errors="coerce").min()

        if "max_score" in group.columns:
            out["max_score"] = pd.to_numeric(group["max_score"], errors="coerce").max()

        return pd.Series(out)

    return (
        data
        .groupby(group_cols, dropna=False)
        .apply(summarize)
        .reset_index()
    )


df = collapse_duplicate_norms(df)

df["norm_grade"] = pd.Categorical(
    df["norm_grade"],
    categories=grade_order,
    ordered=True,
)


# ============================================================
# HELPERS
# ============================================================

def fmt_value(value, metric_col):
    if pd.isna(value):
        return "—"
    if metric_col == "mean_score":
        return f"{value:.2f}"
    return f"{value:.1f}%"


def fmt_number(value):
    if pd.isna(value):
        return "—"
    return f"{value:,.0f}"


def confidence_label(base):
    if pd.isna(base):
        return "Directional", "confidence-directional", "Base unavailable"
    if base >= 500:
        return "Strong", "confidence-strong", "Large base"
    if base >= 100:
        return "Reliable", "confidence-reliable", "Solid base"
    if base >= 30:
        return "Directional", "confidence-reliable", "Use as signal"
    return "Low base", "confidence-directional", "Read carefully"


def get_tier_value(data, grade, metric_col):
    temp = data[data["norm_grade"].astype(str) == grade]
    if temp.empty:
        return np.nan
    return temp[metric_col].mean()


def render_metric(col, label, value, note):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal(col, title, pill, copy):
    col.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-title">{title}</div>
            <div class="signal-copy">
                <span class="pill">{pill}</span><br>
                {copy}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(SIDEBAR_LOGO_HTML, unsafe_allow_html=True)
st.sidebar.markdown("## Filters")
st.sidebar.caption("Choose a benchmark cut, then refine.")

available_slices = df["slice_type"].dropna().unique().tolist()

preferred_slice_order = [
    "Global",
    "category",
    "sub_category",
    "detail_product",
    "gender",
    "age_group",
    "ses",
    "occupation",
    "type_of_study",
    "test_type",
    "methodology",
    "sub_method",
    "num_of_product",
    "sequence",
    "category | gender",
    "category | age_group",
    "category | ses",
    "category | occupation",
    "category | methodology",
    "category | test_type",
    "category | sub_category",
    "category | detail_product",
    "sub_category | gender",
    "detail_product | gender",
    "methodology | test_type",
    "type_of_study | methodology",
]

ordered_slices = [x for x in preferred_slice_order if x in available_slices]
remaining_slices = sorted([x for x in available_slices if x not in ordered_slices])
ordered_slices = ordered_slices + remaining_slices

display_options = [clean_label(x) for x in ordered_slices]
display_to_raw = dict(zip(display_options, ordered_slices))

default_display = "Overall Market" if "Overall Market" in display_options else display_options[0]

selected_display = st.sidebar.selectbox(
    "Benchmark view",
    display_options,
    index=display_options.index(default_display),
)

selected_slice = display_to_raw[selected_display]
filtered = df[df["slice_type"] == selected_slice].copy()

slice_columns = selected_slice.split(" | ") if selected_slice != "Global" else []
slice_columns = [c for c in slice_columns if c in filtered.columns]

for col in slice_columns:
    values = filtered[col].dropna().astype(str).unique().tolist()
    values = sort_dropdown_values(col, values)

    if values:
        chosen = st.sidebar.multiselect(clean_label(col), values, default=[])
        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

parameters = sorted(filtered["parameter_name"].dropna().unique().tolist())
chosen_params = st.sidebar.multiselect("Parameter", parameters, default=[])

if chosen_params:
    filtered = filtered[filtered["parameter_name"].isin(chosen_params)]

scales = filtered["scale"].dropna().unique().tolist()
scales = sort_dropdown_values("scale", scales)

chosen_scales = st.sidebar.multiselect("Scale", scales, default=scales)

if chosen_scales:
    filtered = filtered[filtered["scale"].isin(chosen_scales)]

grades = [
    g for g in grade_order
    if g in filtered["norm_grade"].dropna().astype(str).unique()
]

chosen_grades = st.sidebar.multiselect("Norm group", grades, default=grades)

if chosen_grades:
    filtered = filtered[filtered["norm_grade"].astype(str).isin(chosen_grades)]

min_base = st.sidebar.number_input("Minimum base", min_value=0, value=10, step=10)
filtered = filtered[filtered["base"] >= min_base].copy()

metric_options = {
    "Mean Score": "mean_score",
    "Top Box": "tb_pct",
    "Top 2 Boxes": "t2b_pct",
    "Top 3 Boxes": "t3b_pct",
}

metric_label = st.sidebar.selectbox("Focus metric", list(metric_options.keys()))
metric_col = metric_options[metric_label]

top_n = st.sidebar.slider("Top parameters", min_value=5, max_value=20, value=10)

if filtered.empty:
    st.warning("No benchmark found. Try another filter or lower the minimum base.")
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="brand-row">
            <div class="brand-left">
                {LOGO_HTML}
                <div class="brand-sub">Survey Norm Database</div>
            </div>
            <div class="brand-chip">Insight Benchmark</div>
        </div>

        <div class="eyebrow">DEKA INSIGHT • ANALYTICS TOOL</div>

        <div class="hero-title">
            Score context,<br>
            <span class="hero-accent">not just score tracking.</span>
        </div>

        <div class="hero-copy">
            Read every survey score against historical norms. See what is strong, what is normal,
            and what needs attention — by attribute, segment, scale, and norm tier.
        </div>

        <div class="mini-flow">
            <span><b>01</b> Select cut</span>
            <span><b>02</b> Read norm</span>
            <span><b>03</b> Check score</span>
            <span><b>04</b> Spot gap</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KPI SNAPSHOT
# ============================================================

norm_rows = len(filtered)
median_base = filtered["base"].median()
avg_mean = filtered["mean_score"].mean()
avg_tb = filtered["tb_pct"].mean()
avg_t2b = filtered["t2b_pct"].mean()
avg_t3b = filtered["t3b_pct"].mean(skipna=True)

conf_label, conf_class, conf_note = confidence_label(median_base)

st.markdown('<div class="section-title">Benchmark snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">A compact read of the selected norm cut.</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)

render_metric(k1, "Rows", f"{norm_rows:,}", "Norm rows")
render_metric(k2, "Base", fmt_number(median_base), "Median base")
render_metric(k3, "Mean", f"{avg_mean:.2f}", "Avg score")
render_metric(k4, "TB", f"{avg_tb:.1f}%", "Top Box")
render_metric(k5, "T2B", f"{avg_t2b:.1f}%", "Top 2")
render_metric(k6, "T3B", "—" if pd.isna(avg_t3b) else f"{avg_t3b:.1f}%", "Scale 7+")

st.markdown(
    """
    <div class="hint">
        Compare scores within the same scale only. A 5-point score should not be read against 7-point or 9-point norms.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# VERDICT
# ============================================================

top_value = get_tier_value(filtered, "Top 25%", metric_col)
avg_value = get_tier_value(filtered, "Average 50%", metric_col)
bottom_value = get_tier_value(filtered, "Bottom 25%", metric_col)

st.markdown('<div class="section-title">Norm verdict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">The benchmark line for strong, normal, and weak performance.</div>',
    unsafe_allow_html=True,
)

v1, v2, v3, v4 = st.columns([1.25, 1, 1, 1])

v1.markdown(
    f"""
    <div class="verdict-card">
        <div class="verdict-kicker">Current cut</div>
        <div class="verdict-main">Strong starts at {fmt_value(top_value, metric_col)}</div>
        <div class="verdict-text">
            Use Top 25% as the high benchmark, Average 50% as the norm line,
            and Bottom 25% as the watch-out zone.
            <br>
            <span class="confidence {conf_class}">{conf_label}</span>
            <br>{conf_note}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def tier_card(col, title, value, copy):
    col.markdown(
        f"""
        <div class="tier-card">
            <div class="tier-title">{title}</div>
            <div class="tier-value">{fmt_value(value, metric_col)}</div>
            <div class="tier-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tier_card(v2, "Top 25%", top_value, "Strong benchmark.")
tier_card(v3, "Average 50%", avg_value, "Normal reference.")
tier_card(v4, "Bottom 25%", bottom_value, "Watch-out zone.")

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# SCORE CHECKER
# ============================================================

st.markdown('<div class="section-title">Score checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Input a score and get a quick norm read.</div>',
    unsafe_allow_html=True,
)

pair_tiers = (
    filtered
    .groupby(["parameter_name", "scale"], dropna=False)["norm_grade"]
    .apply(lambda s: set(s.astype(str)))
    .reset_index(name="tiers")
)

pair_tiers["tier_count"] = pair_tiers["tiers"].apply(len)
pair_tiers["has_top_avg"] = pair_tiers["tiers"].apply(
    lambda x: ("Top 25%" in x) and ("Average 50%" in x)
)

pair_tiers = pair_tiers.sort_values(
    ["has_top_avg", "tier_count", "parameter_name", "scale"],
    ascending=[False, False, True, True],
)

pair_tiers["label"] = (
    pair_tiers["parameter_name"].astype(str)
    + " · "
    + pair_tiers["scale"].astype(str)
    + " pts"
)

pair_labels = pair_tiers["label"].tolist()
label_to_pair = pair_tiers.set_index("label")[["parameter_name", "scale"]].to_dict("index")

c1, c2, c3 = st.columns([1.7, 0.75, 1.15])

selected_pair = c1.selectbox("Parameter", pair_labels)
selected_param = label_to_pair[selected_pair]["parameter_name"]
selected_scale = label_to_pair[selected_pair]["scale"]

check_scope = filtered[
    (filtered["parameter_name"] == selected_param)
    & (filtered["scale"] == selected_scale)
].copy()

if metric_col == "mean_score":
    score_max = float(selected_scale)
else:
    score_max = 100.0

default_score = check_scope[metric_col].mean()
if pd.isna(default_score):
    default_score = 0.0

score_value = c2.number_input(
    "Score",
    min_value=0.0,
    max_value=score_max,
    value=float(min(default_score, score_max)),
    step=0.1,
)

top_ref = get_tier_value(check_scope, "Top 25%", metric_col)
avg_ref = get_tier_value(check_scope, "Average 50%", metric_col)
bottom_ref = get_tier_value(check_scope, "Bottom 25%", metric_col)

if pd.notna(top_ref) and pd.notna(avg_ref):
    if score_value >= top_ref:
        score_status = "Above norm"
        score_note = "Strong. Treat as a winning signal."
    elif score_value >= avg_ref:
        score_status = "On norm"
        score_note = "Healthy. Around market reference."
    elif pd.notna(bottom_ref) and score_value <= bottom_ref:
        score_status = "Below norm"
        score_note = "Weak. Needs closer review."
    else:
        score_status = "Below average"
        score_note = "Not critical, but room to improve."

    c3.markdown(
        f"""
        <div class="tier-card" style="min-height: 112px;">
            <div class="tier-title">{score_status}</div>
            <div class="tier-copy">{score_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    c3.markdown(
        """
        <div class="tier-card" style="min-height: 112px;">
            <div class="tier-title">Limited norm</div>
            <div class="tier-copy">This parameter does not have enough tiers for a score read.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# KEY SIGNALS
# ============================================================

rank_df = filtered.dropna(subset=[metric_col]).copy()

best = rank_df.sort_values(metric_col, ascending=False).head(1)
weak = rank_df.sort_values(metric_col, ascending=True).head(1)

gap_source = filtered[
    filtered["norm_grade"].astype(str).isin(["Top 25%", "Bottom 25%"])
].copy()

gap_pivot = gap_source.pivot_table(
    index=["parameter_name", "scale"],
    columns="norm_grade",
    values=metric_col,
    aggfunc="mean",
).reset_index()

gap_driver = pd.DataFrame()

if "Top 25%" in gap_pivot.columns and "Bottom 25%" in gap_pivot.columns:
    gap_pivot["gap"] = gap_pivot["Top 25%"] - gap_pivot["Bottom 25%"]
    gap_driver = gap_pivot.sort_values("gap", ascending=False).head(1)

st.markdown('<div class="section-title">Key signals</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Quick cues from the current benchmark.</div>',
    unsafe_allow_html=True,
)

i1, i2, i3 = st.columns(3)

if not best.empty:
    r = best.iloc[0]
    render_signal(
        i1,
        "Best cue",
        str(r["norm_grade"]),
        f"<b>{r['parameter_name']}</b> is the strongest reference on <b>{metric_label}</b>.",
    )

if not weak.empty:
    r = weak.iloc[0]
    render_signal(
        i2,
        "Watch-out",
        str(r["norm_grade"]),
        f"<b>{r['parameter_name']}</b> sits lowest. Review before calling the concept strong.",
    )

if not gap_driver.empty:
    r = gap_driver.iloc[0]
    render_signal(
        i3,
        "Gap driver",
        "Top vs Bottom",
        f"<b>{r['parameter_name']}</b> separates strong and weak results the most.",
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# CHARTS
# ============================================================

st.markdown('<div class="section-title">Benchmark views</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Ranking, tier comparison, and spread.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.25, 1])

ranking_tier = "Top 25%" if "Top 25%" in filtered["norm_grade"].astype(str).unique() else filtered["norm_grade"].astype(str).iloc[0]

rank_chart = (
    filtered[filtered["norm_grade"].astype(str) == ranking_tier]
    .dropna(subset=[metric_col])
    .groupby(["parameter_name", "scale"], dropna=False)
    .agg(value=(metric_col, "mean"), base=("base", "sum"))
    .reset_index()
    .sort_values("value", ascending=False)
    .head(top_n)
)

with left:
    st.markdown('<div class="section-title" style="font-size:1.20rem;">Top benchmark ranking</div>', unsafe_allow_html=True)

    if rank_chart.empty:
        st.info("No ranking available.")
    else:
        fig = px.bar(
            rank_chart,
            x="value",
            y="parameter_name",
            orientation="h",
            color_discrete_sequence=[GOLD],
            hover_data=["scale", "base"],
        )

        fig.update_layout(
            height=470,
            plot_bgcolor=CREAM,
            paper_bgcolor=CREAM,
            font=dict(color=NAVY, size=12),
            xaxis_title=metric_label,
            yaxis_title="",
            margin=dict(l=0, r=10, t=10, b=10),
        )

        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

with right:
    st.markdown('<div class="section-title" style="font-size:1.20rem;">Norm tier comparison</div>', unsafe_allow_html=True)

    grade_df = (
        filtered
        .dropna(subset=[metric_col])
        .groupby("norm_grade", observed=False)
        .agg(value=(metric_col, "mean"), base=("base", "sum"))
        .reset_index()
        .dropna(subset=["value"])
    )

    if grade_df.empty:
        st.info("No tier comparison available.")
    else:
        fig2 = px.bar(
            grade_df,
            x="norm_grade",
            y="value",
            color="norm_grade",
            text="value",
            color_discrete_map={
                "Top 25%": GOLD,
                "Average 50%": BLUE,
                "Bottom 25%": TAUPE,
            },
        )

        fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")

        fig2.update_layout(
            height=470,
            plot_bgcolor=CREAM,
            paper_bgcolor=CREAM,
            font=dict(color=NAVY, size=12),
            showlegend=False,
            xaxis_title="",
            yaxis_title=metric_label,
            margin=dict(l=0, r=10, t=10, b=10),
        )

        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# GAP ANALYSIS
# ============================================================

st.markdown('<div class="section-title">Top–Bottom gap</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">The attributes that separate strong from weak results.</div>',
    unsafe_allow_html=True,
)

if "Top 25%" in gap_pivot.columns and "Bottom 25%" in gap_pivot.columns:
    gap_plot = gap_pivot.copy()
    gap_plot["gap"] = gap_plot["Top 25%"] - gap_plot["Bottom 25%"]
    gap_plot = gap_plot.sort_values("gap", ascending=False).head(top_n)

    fig3 = px.bar(
        gap_plot,
        x="gap",
        y="parameter_name",
        orientation="h",
        color="gap",
        color_continuous_scale=[[0, "#EDE6DA"], [1, GOLD]],
        hover_data=["scale", "Top 25%", "Bottom 25%"],
    )

    fig3.update_layout(
        height=430,
        plot_bgcolor=CREAM,
        paper_bgcolor=CREAM,
        font=dict(color=NAVY, size=12),
        xaxis_title=f"Gap in {metric_label}",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10),
    )

    fig3.update_yaxes(autorange="reversed")

    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

else:
    st.info("Select both Top 25% and Bottom 25% to see the gap view.")

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# READING NOTES
# ============================================================

st.markdown('<div class="section-title">Reading notes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Keep the benchmark useful and honest.</div>',
    unsafe_allow_html=True,
)

n1, n2, n3 = st.columns(3)

render_signal(
    n1,
    "Same scale only",
    "Method",
    "Read each score against the same scale. Do not mix 5, 7, and 9-point norms.",
)

render_signal(
    n2,
    "Use the gap",
    "Priority",
    "Wide Top–Bottom gaps point to stronger optimization levers.",
)

render_signal(
    n3,
    "Check the base",
    conf_label,
    f"Current read: <b>{conf_note}</b>. Low-base cuts are directional.",
)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# TABLE
# ============================================================

st.markdown('<div class="section-title">Norm table</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Clean export table for reporting.</div>',
    unsafe_allow_html=True,
)

base_cols = [
    "parameter_name",
    "scale",
    "norm_grade",
    "mean_score",
    "tb_pct",
    "t2b_pct",
    "t3b_pct",
    "base",
    "min_score",
    "max_score",
    "std_score",
]

segment_cols = [
    "category",
    "sub_category",
    "detail_product",
    "gender",
    "age_group",
    "actual_age",
    "ses",
    "occupation",
    "type_of_study",
    "test_type",
    "methodology",
    "sub_method",
    "num_of_product",
    "sequence",
]

visible_segment_cols = [
    col for col in segment_cols
    if col in filtered.columns and filtered[col].notna().any()
]

display_cols = visible_segment_cols + [c for c in base_cols if c in filtered.columns]
table_df = filtered[display_cols].copy()

rename_cols = {
    "parameter_name": "Parameter",
    "scale": "Scale",
    "norm_grade": "Norm Group",
    "mean_score": "Mean",
    "tb_pct": "TB%",
    "t2b_pct": "T2B%",
    "t3b_pct": "T3B%",
    "base": "Base",
    "min_score": "Min",
    "max_score": "Max",
    "std_score": "Std",
    "category": "Category",
    "sub_category": "Sub Category",
    "detail_product": "Detail Product",
    "gender": "Gender",
    "age_group": "Age Group",
    "actual_age": "Age",
    "ses": "SES",
    "occupation": "Occupation",
    "type_of_study": "Type of Study",
    "test_type": "Test Type",
    "methodology": "Methodology",
    "sub_method": "Sub Method",
    "num_of_product": "# Product",
    "sequence": "Sequence",
}

table_df = table_df.rename(columns=rename_cols)

sort_cols = [c for c in ["Parameter", "Scale", "Norm Group"] if c in table_df.columns]

if sort_cols:
    table_df = table_df.sort_values(sort_cols)

table_df = table_df.astype("object")
table_df = table_df.where(pd.notna(table_df), "—")
table_df = table_df.replace(["None", "nan", "NaN", "NULL", ""], "—")

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
    height=430,
)

csv = table_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "deka_norm_filtered.csv",
    "text/csv",
)

st.markdown(
    """
    <div class="hint">
        Norm groups are calculated from ranked respondent-level scores within each parameter and scale.
        Dashboard reads are for benchmark interpretation, not causal inference.
    </div>
    """,
    unsafe_allow_html=True,
)
