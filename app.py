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
# LOGO & PAGE ICON
# ============================================================

APP_ICON_PATH = Path("deka-icon.png")
LOGO_PATH = Path("deka-logo.png")

if APP_ICON_PATH.exists():
    PAGE_ICON = Image.open(APP_ICON_PATH)
elif LOGO_PATH.exists():
    PAGE_ICON = Image.open(LOGO_PATH)
else:
    PAGE_ICON = "🟡"


# ============================================================
# PAGE CONFIG
# Must be the first Streamlit command
# ============================================================

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon=PAGE_ICON,
    layout="wide"
)


# ============================================================
# BRAND PALETTE
# ============================================================

NAVY = "#0B1026"
BLUE = "#1E2A4A"
GOLD = "#F2A93B"
CREAM = "#FAF8F2"
CARD = "#FFFFFF"
GREY = "#747B8D"
LINE = "#E7E0D6"
GREEN = "#2E7D5B"
RED = "#D95F59"
TAUPE = "#C8BFB2"


# ============================================================
# LOGO HANDLING
# ============================================================

def image_to_base64(path):
    path = Path(path)

    if not path.exists():
        return None

    return base64.b64encode(path.read_bytes()).decode()


logo_base64 = image_to_base64(LOGO_PATH)

if logo_base64:
    LOGO_HTML = f"""
        <img src="data:image/png;base64,{logo_base64}" class="brand-logo" />
    """
else:
    LOGO_HTML = """
        <div class="brand-fallback">
            <span>deka</span><br>insight
        </div>
    """


# ============================================================
# GLOBAL STYLE
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
                radial-gradient(circle at 12% 8%, rgba(242,169,59,0.12), transparent 24%),
                linear-gradient(180deg, #FFFCF7 0%, {CREAM} 100%);
            color: {NAVY};
        }}

        .main .block-container {{
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #ECE7DD;
        }}

        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p {{
            color: {BLUE};
        }}

        .brand-bar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 26px;
            position: relative;
            z-index: 2;
        }}

        .brand-left {{
            display: flex;
            align-items: center;
            gap: 14px;
        }}

        .brand-logo {{
            width: 96px;
            height: auto;
            object-fit: contain;
        }}

        .brand-fallback {{
            color: {NAVY};
            font-weight: 900;
            line-height: 0.9;
            font-size: 1.15rem;
        }}

        .brand-fallback span {{
            color: {GOLD};
        }}

        .brand-kicker {{
            color: {GREY};
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.04em;
        }}

        .brand-pill {{
            background: {NAVY};
            color: white;
            padding: 10px 18px;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 800;
        }}

        .hero {{
            position: relative;
            background:
                radial-gradient(circle at top left, rgba(242,169,59,0.20), transparent 30%),
                linear-gradient(135deg, #FFFFFF 0%, #FFF8EC 100%);
            border: 1px solid #EFE1CB;
            border-radius: 34px;
            padding: 42px 52px 46px 52px;
            box-shadow: 0 28px 70px rgba(11,16,38,0.08);
            margin-bottom: 34px;
            overflow: hidden;
        }}

        .hero:after {{
            content: "";
            position: absolute;
            right: -90px;
            top: -90px;
            width: 280px;
            height: 280px;
            background: rgba(242,169,59,0.13);
            border-radius: 50%;
        }}

        .eyebrow {{
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            color: {GOLD};
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 16px;
            position: relative;
            z-index: 2;
        }}

        .hero-title {{
            font-size: 3.25rem;
            line-height: 1.02;
            color: {NAVY};
            font-weight: 900;
            letter-spacing: -0.055em;
            margin-bottom: 18px;
            max-width: 980px;
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
            font-size: 1.06rem;
            line-height: 1.75;
            color: {BLUE};
            max-width: 930px;
            position: relative;
            z-index: 2;
        }}

        .method-strip {{
            display: flex;
            gap: 12px;
            margin-top: 26px;
            flex-wrap: wrap;
            position: relative;
            z-index: 2;
        }}

        .method-item {{
            background: rgba(255,255,255,0.75);
            border: 1px solid #EFE1CB;
            border-radius: 999px;
            padding: 9px 14px;
            color: {BLUE};
            font-size: 0.84rem;
            font-weight: 800;
        }}

        .method-item span {{
            color: {GOLD};
        }}

        .section-title {{
            color: {NAVY};
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -0.035em;
            margin: 8px 0 6px 0;
        }}

        .section-subtitle {{
            color: {GREY};
            font-size: 0.96rem;
            margin-bottom: 18px;
        }}

        .metric-card {{
            background: {CARD};
            border: 1px solid {LINE};
            border-radius: 24px;
            padding: 22px 22px;
            box-shadow: 0 14px 32px rgba(11,16,38,0.06);
            height: 148px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}

        .metric-label {{
            font-size: 0.73rem;
            color: {GREY};
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: {NAVY};
            font-size: 2.02rem;
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1;
        }}

        .metric-note {{
            color: {GREY};
            font-size: 0.80rem;
            line-height: 1.35;
        }}

        .verdict-card {{
            background: {NAVY};
            color: white;
            border-radius: 28px;
            padding: 28px 30px;
            box-shadow: 0 22px 52px rgba(11,16,38,0.18);
            min-height: 215px;
        }}

        .verdict-label {{
            color: {GOLD};
            font-size: 0.75rem;
            font-weight: 900;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 14px;
        }}

        .verdict-title {{
            color: white;
            font-size: 1.65rem;
            font-weight: 900;
            letter-spacing: -0.035em;
            line-height: 1.18;
            margin-bottom: 12px;
        }}

        .verdict-copy {{
            color: #E6E9F0;
            font-size: 0.96rem;
            line-height: 1.6;
        }}

        .threshold-card {{
            background: white;
            border: 1px solid {LINE};
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 14px 32px rgba(11,16,38,0.06);
            min-height: 215px;
        }}

        .threshold-title {{
            color: {NAVY};
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 10px;
        }}

        .threshold-value {{
            color: {GOLD};
            font-size: 2.15rem;
            line-height: 1;
            font-weight: 900;
            letter-spacing: -0.045em;
            margin-bottom: 10px;
        }}

        .threshold-copy {{
            color: {BLUE};
            font-size: 0.92rem;
            line-height: 1.55;
        }}

        .insight-card {{
            background: {CARD};
            border: 1px solid {LINE};
            border-left: 7px solid {GOLD};
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 16px 34px rgba(11,16,38,0.06);
            min-height: 150px;
        }}

        .insight-title {{
            color: {NAVY};
            font-size: 1.05rem;
            font-weight: 900;
            margin-bottom: 8px;
            letter-spacing: -0.02em;
        }}

        .insight-copy {{
            color: {BLUE};
            font-size: 0.96rem;
            line-height: 1.55;
        }}

        .pill {{
            display: inline-block;
            background: rgba(242,169,59,0.16);
            color: {NAVY};
            border: 1px solid rgba(242,169,59,0.36);
            padding: 4px 10px;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 800;
            margin-bottom: 10px;
        }}

        .confidence {{
            display: inline-block;
            border-radius: 999px;
            padding: 5px 11px;
            font-size: 0.78rem;
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

        .score-box {{
            background: white;
            border: 1px solid {LINE};
            border-radius: 26px;
            padding: 24px 26px;
            box-shadow: 0 14px 34px rgba(11,16,38,0.05);
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 20px;
            overflow: hidden;
            border: 1px solid {LINE};
            box-shadow: 0 14px 34px rgba(11,16,38,0.05);
        }}

        .stDownloadButton > button {{
            background: {GOLD};
            color: {NAVY};
            border: none;
            border-radius: 999px;
            padding: 0.72rem 1.25rem;
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
            margin: 34px 0;
        }}

        .small-note {{
            color: {GREY};
            font-size: 0.86rem;
            line-height: 1.5;
        }}
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE CONNECTION
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
# DATA CLEANING
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
# DISPLAY LABELS & SORTING
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
# COLLAPSE DUPLICATE DISPLAY ROWS
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
# GENERAL HELPERS
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
        return "Directional", "confidence-directional", "Base not available."

    if base >= 500:
        return "Strong", "confidence-strong", "Large base. Good for benchmark reading."

    if base >= 100:
        return "Reliable", "confidence-reliable", "Enough base for comparison."

    if base >= 30:
        return "Directional", "confidence-reliable", "Useful signal, but read with care."

    return "Low base", "confidence-directional", "Too small for a firm conclusion."


def get_tier_value(data, grade, metric_col):
    temp = data[data["norm_grade"].astype(str) == grade]

    if temp.empty:
        return np.nan

    return temp[metric_col].mean()


# ============================================================
# SIDEBAR
# ============================================================

if LOGO_PATH.exists():
    st.sidebar.image(str(LOGO_PATH), width=105)

st.sidebar.markdown("## Filters")
st.sidebar.caption("Pick a benchmark cut and refine the comparison.")

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
        chosen = st.sidebar.multiselect(
            clean_label(col),
            values,
            default=[],
        )

        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

parameters = sorted(filtered["parameter_name"].dropna().unique().tolist())

chosen_params = st.sidebar.multiselect(
    "Parameter",
    parameters,
    default=[],
)

if chosen_params:
    filtered = filtered[filtered["parameter_name"].isin(chosen_params)]

scales = filtered["scale"].dropna().unique().tolist()
scales = sort_dropdown_values("scale", scales)

chosen_scales = st.sidebar.multiselect(
    "Scale",
    scales,
    default=scales,
)

if chosen_scales:
    filtered = filtered[filtered["scale"].isin(chosen_scales)]

grades = [
    g for g in grade_order
    if g in filtered["norm_grade"].dropna().astype(str).unique()
]

chosen_grades = st.sidebar.multiselect(
    "Norm group",
    grades,
    default=grades,
)

if chosen_grades:
    filtered = filtered[filtered["norm_grade"].astype(str).isin(chosen_grades)]

min_base = st.sidebar.number_input(
    "Minimum base",
    min_value=0,
    value=10,
    step=10,
)

filtered = filtered[filtered["base"] >= min_base].copy()

metric_options = {
    "Mean Score": "mean_score",
    "Top Box": "tb_pct",
    "Top 2 Boxes": "t2b_pct",
    "Top 3 Boxes": "t3b_pct",
}

metric_label = st.sidebar.selectbox(
    "Focus metric",
    list(metric_options.keys()),
)

metric_col = metric_options[metric_label]

top_n = st.sidebar.slider(
    "Show top parameters",
    min_value=5,
    max_value=25,
    value=12,
)

if filtered.empty:
    st.warning("No benchmark found for this selection. Try lowering the base or changing the segment.")
    st.stop()


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="brand-bar">
            <div class="brand-left">
                {LOGO_HTML}
                <div class="brand-kicker">Survey Norm Database</div>
            </div>
            <div class="brand-pill">Consumer Insight Benchmark</div>
        </div>

        <div class="eyebrow">DEKA INSIGHT • ANALYTICS TOOL</div>

        <div class="hero-title">
            Know whether a score is<br>
            <span class="hero-accent">good, average, or below norm.</span>
        </div>

        <div class="hero-copy">
            A benchmark dashboard for reading survey scores against Deka’s historical database.
            Compare every attribute by segment, scale, and norm tier — then turn raw scores into sharper business calls.
        </div>

        <div class="method-strip">
            <div class="method-item"><span>01</span> Choose segment</div>
            <div class="method-item"><span>02</span> Read the norm</div>
            <div class="method-item"><span>03</span> Spot the gap</div>
            <div class="method-item"><span>04</span> Decide the next move</div>
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

st.markdown('<div class="section-title">Norm at a glance</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">The fastest read of the selected benchmark.</div>',
    unsafe_allow_html=True,
)

k1, k2, k3, k4, k5, k6 = st.columns(6)


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


render_metric(k1, "Norm Rows", f"{norm_rows:,}", "Benchmark rows")
render_metric(k2, "Median Base", fmt_number(median_base), "Typical sample size")
render_metric(k3, "Mean", f"{avg_mean:.2f}", "Average score")
render_metric(k4, "TB", f"{avg_tb:.1f}%", "Top Box")
render_metric(k5, "T2B", f"{avg_t2b:.1f}%", "Top 2 Boxes")
render_metric(k6, "T3B", "—" if pd.isna(avg_t3b) else f"{avg_t3b:.1f}%", "Scale 7+")

st.markdown(
    """
    <p class="small-note">
        Scores are benchmarked within the same scale. Avoid comparing 5-point, 7-point, and 9-point results directly.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# BENCHMARK VERDICT
# ============================================================

top_value = get_tier_value(filtered, "Top 25%", metric_col)
avg_value = get_tier_value(filtered, "Average 50%", metric_col)
bottom_value = get_tier_value(filtered, "Bottom 25%", metric_col)

st.markdown('<div class="section-title">Benchmark verdict</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">What strong, typical, and weak performance look like in this cut.</div>',
    unsafe_allow_html=True,
)

v1, v2, v3, v4 = st.columns([1.35, 1, 1, 1])

v1.markdown(
    f"""
    <div class="verdict-card">
        <div class="verdict-label">Current benchmark</div>
        <div class="verdict-title">
            Strong performance starts around {fmt_value(top_value, metric_col)}.
        </div>
        <div class="verdict-copy">
            Top 25% is the upper reference for {metric_label}. Average 50% is the practical baseline,
            while Bottom 25% marks the watch-out range.
            <br>
            <span class="confidence {conf_class}">{conf_label}</span>
            <br><br>{conf_note}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def threshold_card(col, title, value, copy):
    col.markdown(
        f"""
        <div class="threshold-card">
            <div class="threshold-title">{title}</div>
            <div class="threshold-value">{fmt_value(value, metric_col)}</div>
            <div class="threshold-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


threshold_card(
    v2,
    "Top 25%",
    top_value,
    "Winning benchmark. Scores close to this range are strong.",
)

threshold_card(
    v3,
    "Average 50%",
    avg_value,
    "Normal zone. Use this as the baseline for comparison.",
)

threshold_card(
    v4,
    "Bottom 25%",
    bottom_value,
    "Watch-out range. Scores here need closer diagnosis.",
)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# SCORE CHECKER
# ============================================================

st.markdown('<div class="section-title">Score check</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Input a product score and see where it lands against the selected norm.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="score-box">', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1.5, 0.8, 0.9, 1.2])

check_params = sorted(filtered["parameter_name"].dropna().unique().tolist())
check_param = c1.selectbox("Parameter to check", check_params)

check_scope = filtered[filtered["parameter_name"] == check_param].copy()

check_scales = sort_dropdown_values("scale", check_scope["scale"].dropna().unique().tolist())
check_scale = c2.selectbox("Scale", check_scales)

check_scope = check_scope[check_scope["scale"] == check_scale].copy()

if metric_col == "mean_score":
    max_score = float(check_scale)
    default_score = float(check_scope[metric_col].mean()) if not check_scope.empty else 0.0
else:
    max_score = 100.0
    default_score = float(check_scope[metric_col].mean()) if not check_scope.empty else 0.0

if pd.isna(default_score):
    default_score = 0.0

score_value = c3.number_input(
    "Your score",
    min_value=0.0,
    max_value=max_score,
    value=min(default_score, max_score),
    step=0.1,
)

top_ref = get_tier_value(check_scope, "Top 25%", metric_col)
avg_ref = get_tier_value(check_scope, "Average 50%", metric_col)
bottom_ref = get_tier_value(check_scope, "Bottom 25%", metric_col)

if pd.notna(top_ref) and pd.notna(avg_ref):
    if score_value >= top_ref:
        verdict = "Above top norm"
        verdict_note = "Strong result. This score sits in the winning range."
    elif score_value >= avg_ref:
        verdict = "Within norm"
        verdict_note = "Healthy result. This score is around the market baseline."
    elif pd.notna(bottom_ref) and score_value <= bottom_ref:
        verdict = "Below norm"
        verdict_note = "Weak result. This score needs attention."
    else:
        verdict = "Between average and bottom"
        verdict_note = "Not critical, but there is room to improve."

    c4.markdown(
        f"""
        <div class="threshold-card" style="min-height: 120px; padding: 18px 20px;">
            <div class="threshold-title">{verdict}</div>
            <div class="threshold-copy">{verdict_note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    c4.info("Not enough norm tiers for this parameter.")

st.markdown("</div>", unsafe_allow_html=True)

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
    '<div class="section-subtitle">Short reads for sharper discussion.</div>',
    unsafe_allow_html=True,
)

i1, i2, i3 = st.columns(3)

if not best.empty:
    r = best.iloc[0]
    i1.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Best cue</div>
            <div class="insight-copy">
                <span class="pill">{r['norm_grade']}</span><br>
                <b>{r['parameter_name']}</b> is the strongest reference point on <b>{metric_label}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not weak.empty:
    r = weak.iloc[0]
    i2.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Watch-out</div>
            <div class="insight-copy">
                <span class="pill">{r['norm_grade']}</span><br>
                <b>{r['parameter_name']}</b> sits lowest. Review this before calling the concept strong.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not gap_driver.empty:
    r = gap_driver.iloc[0]
    i3.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Gap driver</div>
            <div class="insight-copy">
                <span class="pill">Top vs Bottom</span><br>
                <b>{r['parameter_name']}</b> separates strong and weak results the most.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    i3.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Current cut</div>
            <div class="insight-copy">
                <span class="pill">{clean_label(selected_slice)}</span><br>
                <b>{len(filtered):,}</b> norm rows with minimum base <b>{min_base}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# CHART DATA
# ============================================================

chart_df = (
    filtered
    .dropna(subset=[metric_col])
    .groupby(["parameter_name", "scale", "norm_grade"], observed=False, dropna=False)
    .agg(
        mean_score=("mean_score", "mean"),
        tb_pct=("tb_pct", "mean"),
        t2b_pct=("t2b_pct", "mean"),
        t3b_pct=("t3b_pct", "mean"),
        base=("base", "sum"),
    )
    .reset_index()
)

chart_df = chart_df.sort_values(metric_col, ascending=False).head(top_n * 3)


# ============================================================
# CHARTS
# ============================================================

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-title">Attribute benchmark ranking</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">Top attributes by {metric_label}.</div>',
        unsafe_allow_html=True,
    )

    if chart_df.empty:
        st.info("No chart available for this metric and filter.")
    else:
        fig = px.bar(
            chart_df,
            x=metric_col,
            y="parameter_name",
            color="norm_grade",
            orientation="h",
            hover_data=["scale", "base", "mean_score", "tb_pct", "t2b_pct", "t3b_pct"],
            color_discrete_map={
                "Top 25%": GOLD,
                "Average 50%": BLUE,
                "Bottom 25%": TAUPE,
            },
        )

        fig.update_layout(
            height=540,
            plot_bgcolor=CREAM,
            paper_bgcolor=CREAM,
            font=dict(color=NAVY, size=13),
            legend_title_text="",
            xaxis_title=metric_label,
            yaxis_title="",
            margin=dict(l=0, r=10, t=10, b=10),
        )

        fig.update_yaxes(autorange="reversed")

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False},
        )

with right:
    st.markdown('<div class="section-title">Norm tier comparison</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">Average {metric_label} across norm tiers.</div>',
        unsafe_allow_html=True,
    )

    grade_df = (
        filtered
        .dropna(subset=[metric_col])
        .groupby("norm_grade", observed=False)
        .agg(value=(metric_col, "mean"), base=("base", "sum"))
        .reset_index()
        .dropna(subset=["value"])
    )

    if grade_df.empty:
        st.info("No tier comparison available for this metric.")
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
            height=540,
            plot_bgcolor=CREAM,
            paper_bgcolor=CREAM,
            font=dict(color=NAVY, size=13),
            showlegend=False,
            xaxis_title="",
            yaxis_title=metric_label,
            margin=dict(l=0, r=10, t=10, b=10),
        )

        st.plotly_chart(
            fig2,
            use_container_width=True,
            config={"displayModeBar": False},
        )

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# GAP ANALYSIS
# ============================================================

st.markdown('<div class="section-title">What separates strong from weak</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Attributes with the widest gap between Top 25% and Bottom 25%.</div>',
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
        height=460,
        plot_bgcolor=CREAM,
        paper_bgcolor=CREAM,
        font=dict(color=NAVY, size=13),
        xaxis_title=f"Gap in {metric_label}",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10),
    )

    fig3.update_yaxes(autorange="reversed")

    st.plotly_chart(
        fig3,
        use_container_width=True,
        config={"displayModeBar": False},
    )

else:
    st.info("Select both Top 25% and Bottom 25% to see the gap view.")

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# SO WHAT
# ============================================================

st.markdown('<div class="section-title">So what?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">How to read the benchmark without overclaiming.</div>',
    unsafe_allow_html=True,
)

s1, s2, s3 = st.columns(3)

s1.markdown(
    """
    <div class="insight-card">
        <div class="insight-title">Compare within scale</div>
        <div class="insight-copy">
            Keep 5-point, 7-point, and 9-point scores separate. Norms are meaningful only within the same scale.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

s2.markdown(
    """
    <div class="insight-card">
        <div class="insight-title">Prioritize big gaps</div>
        <div class="insight-copy">
            Attributes with wide Top–Bottom gaps are stronger optimization levers than flat attributes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

s3.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Respect the base</div>
        <div class="insight-copy">
            Current confidence: <b>{conf_label}</b>. Use low-base cuts as signals, not final conclusions.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<hr>", unsafe_allow_html=True)


# ============================================================
# CLEAN TABLE
# ============================================================

st.markdown('<div class="section-title">Export-ready norm table</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Clean benchmark table for review, reporting, or export.</div>',
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

visible_segment_cols = []

for col in segment_cols:
    if col in filtered.columns and filtered[col].notna().any():
        visible_segment_cols.append(col)

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
    <p class="small-note">
        Norm groups are calculated from ranked respondent-level scores within each parameter and scale.
        Dashboard reads are intended for benchmark interpretation, not causal inference.
    </p>
    """,
    unsafe_allow_html=True,
)
