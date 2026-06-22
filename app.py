# ============================================================
# DEKA INSIGHT — NORM DASHBOARD
# Streamlit app.py
# ============================================================

import base64
from pathlib import Path
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine


# ============================================================
# 1. PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. BASIC CONFIG
# ============================================================

APP_TITLE = "Deka Norm Dashboard"
APP_SUBTITLE = "Historical benchmark explorer for product test performance."

BASE_DIR = Path(__file__).parent

LOGO_PATH = BASE_DIR / "deka-logo.png"
ICON_PATH = BASE_DIR / "deka-icon.png"


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def image_to_base64(path):
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def safe_number(x, decimals=1):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):,.{decimals}f}"
    except Exception:
        return "—"


def safe_int(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"


def clean_display_value(x):
    if pd.isna(x):
        return "—"
    x = str(x).strip()
    if x.lower() in ["", "nan", "none", "null", "<na>"]:
        return "—"
    return x


def as_percent(x):
    if pd.isna(x):
        return "—"
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "—"


def filter_options(df, col):
    if col not in df.columns:
        return []

    values = (
        df[col]
        .dropna()
        .astype(str)
        .replace(["nan", "None", "NULL", "<NA>"], np.nan)
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return values


def apply_multiselect_filter(df, col, selected_values):
    if col not in df.columns:
        return df

    if not selected_values:
        return df

    return df[df[col].astype(str).isin(selected_values)].copy()


def metric_column_from_label(label):
    mapping = {
        "Mean Score": "mean_score",
        "Top Box": "tb_pct",
        "Top 2 Boxes": "t2b_pct",
        "Top 3 Boxes": "t3b_pct",
    }
    return mapping.get(label, "mean_score")


def metric_display_name(metric_col):
    mapping = {
        "mean_score": "Mean Score",
        "tb_pct": "Top Box",
        "t2b_pct": "Top 2 Boxes",
        "t3b_pct": "Top 3 Boxes",
    }
    return mapping.get(metric_col, metric_col)


def metric_suffix(metric_col):
    if metric_col in ["tb_pct", "t2b_pct", "t3b_pct"]:
        return "%"
    return ""


def confidence_label(base_value):
    if pd.isna(base_value):
        return "No base"
    if base_value >= 500:
        return "Strong base"
    if base_value >= 100:
        return "Reliable base"
    if base_value >= 30:
        return "Directional"
    return "Low base"


def confidence_class(base_value):
    if pd.isna(base_value):
        return "low"
    if base_value >= 100:
        return "good"
    if base_value >= 30:
        return "medium"
    return "low"


def readable_slice_label(slice_type):
    label_map = {
        "Global": "Overall Market",

        "study": "Study / Project",
        "category": "Category",
        "sub_category": "Sub Category",
        "detail_product": "Detail Product",
        "gender": "Gender",
        "age_group": "Age Group",
        "actual_age": "Actual Age",
        "ses": "SES",
        "occupation": "Occupation",
        "type_of_study": "Type of Study",
        "test_type": "Test Type",
        "methodology": "Methodology",
        "sub_method": "Sub Method",
        "num_of_product": "# of Product",
        "sequence": "Sequence",

        "study | category": "Study × Category",
        "study | gender": "Study × Gender",
        "study | age_group": "Study × Age Group",
        "study | ses": "Study × SES",
        "study | methodology": "Study × Methodology",
        "study | test_type": "Study × Test Type",

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

    return label_map.get(slice_type, slice_type.replace("_", " ").title())


def slice_dimensions(slice_type):
    if slice_type == "Global":
        return []

    return [x.strip() for x in slice_type.split("|")]


def display_col_name(col):
    mapping = {
        "study": "Study / Project",
        "category": "Category",
        "sub_category": "Sub Category",
        "detail_product": "Detail Product",
        "gender": "Gender",
        "age_group": "Age Group",
        "actual_age": "Actual Age",
        "ses": "SES",
        "occupation": "Occupation",
        "type_of_study": "Type of Study",
        "test_type": "Test Type",
        "methodology": "Methodology",
        "sub_method": "Sub Method",
        "num_of_product": "# of Product",
        "sequence": "Sequence",
        "parameter_id": "Parameter ID",
        "parameter_name": "Parameter",
        "parameter_key": "Parameter Key",
        "parameter_group": "Parameter Group",
        "scale": "Scale",
        "norm_grade": "Norm Group",
        "mean_score": "Mean Score",
        "tb_pct": "Top Box (%)",
        "t2b_pct": "Top 2 Boxes (%)",
        "t3b_pct": "Top 3 Boxes (%)",
        "base": "Base",
        "min_score": "Min Score",
        "max_score": "Max Score",
        "std_score": "Std. Dev.",
        "slice_type": "Benchmark View",
    }

    return mapping.get(col, col.replace("_", " ").title())


def make_download_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# ============================================================
# 4. CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #0B1026;
        --blue: #1E2A4A;
        --gold: #F2A93B;
        --cream: #FAF8F2;
        --white: #FFFFFF;
        --muted: #747B8D;
        --line: #E7E0D6;
        --green: #2E7D5B;
        --red: #D95F59;
        --soft-gold: #FFF4DE;
        --soft-blue: #F2F5FB;
    }

    html, body, [class*="css"] {
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #FAF8F2 0%, #FFFFFF 38%, #FAF8F2 100%);
    }

    section[data-testid="stSidebar"] {
        background: #0B1026;
    }

    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #0B1026 !important;
    }

    section[data-testid="stSidebar"] input {
        color: #0B1026 !important;
    }

    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    .hero {
        background: linear-gradient(135deg, #0B1026 0%, #1E2A4A 72%, #2A3558 100%);
        padding: 34px 38px;
        border-radius: 28px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0px 16px 40px rgba(11, 16, 38, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .hero-kicker {
        color: #F2A93B;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        font-size: 42px;
        line-height: 1.05;
        font-weight: 850;
        margin-bottom: 12px;
    }

    .hero-subtitle {
        font-size: 16px;
        line-height: 1.65;
        color: rgba(255,255,255,0.82);
        max-width: 820px;
    }

    .hero-pill {
        display: inline-block;
        padding: 8px 13px;
        border-radius: 999px;
        background: rgba(242, 169, 59, 0.12);
        color: #F2A93B;
        border: 1px solid rgba(242, 169, 59, 0.35);
        font-size: 12px;
        font-weight: 700;
        margin-right: 8px;
        margin-top: 16px;
    }

    .section-title {
        font-size: 21px;
        font-weight: 850;
        color: #0B1026;
        margin: 24px 0 12px 0;
    }

    .section-caption {
        color: #747B8D;
        font-size: 14px;
        margin-top: -6px;
        margin-bottom: 12px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E7E0D6;
        border-radius: 22px;
        padding: 20px 20px;
        box-shadow: 0 10px 28px rgba(11, 16, 38, 0.06);
        min-height: 130px;
    }

    .metric-label {
        color: #747B8D;
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: .08em;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #0B1026;
        font-size: 31px;
        font-weight: 850;
        margin-bottom: 5px;
    }

    .metric-note {
        color: #747B8D;
        font-size: 13px;
    }

    .insight-card {
        background: #FFFFFF;
        border: 1px solid #E7E0D6;
        border-radius: 22px;
        padding: 22px 22px;
        box-shadow: 0 10px 28px rgba(11, 16, 38, 0.06);
        height: 100%;
    }

    .insight-title {
        font-size: 15px;
        font-weight: 850;
        color: #0B1026;
        margin-bottom: 8px;
    }

    .insight-main {
        font-size: 25px;
        font-weight: 850;
        color: #0B1026;
        margin-bottom: 6px;
    }

    .insight-sub {
        font-size: 13px;
        color: #747B8D;
        line-height: 1.5;
    }

    .badge {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 800;
        margin-top: 6px;
    }

    .badge-good {
        background: #EAF6F0;
        color: #2E7D5B;
    }

    .badge-medium {
        background: #FFF4DE;
        color: #B26B00;
    }

    .badge-low {
        background: #FDEEEE;
        color: #D95F59;
    }

    .small-note {
        color: #747B8D;
        font-size: 13px;
        line-height: 1.55;
    }

    .divider {
        border-top: 1px solid #E7E0D6;
        margin: 18px 0;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #E7E0D6;
        font-weight: 800;
    }

    .stDownloadButton > button {
        border-radius: 999px;
        background: #0B1026;
        color: #FFFFFF;
        border: 1px solid #0B1026;
        font-weight: 800;
    }

    .stDownloadButton > button:hover {
        background: #1E2A4A;
        color: #FFFFFF;
        border: 1px solid #1E2A4A;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 5. DATABASE CONNECTION
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine():
    try:
        user = st.secrets["postgres"]["user"]
        password = quote_plus(st.secrets["postgres"]["password"])
        host = st.secrets["postgres"]["host"]
        port = st.secrets["postgres"]["port"]
        database = st.secrets["postgres"]["database"]

        url = (
            f"postgresql+psycopg2://{user}:{password}"
            f"@{host}:{port}/{database}"
            f"?sslmode=require"
        )

        return create_engine(url)

    except Exception as e:
        st.error("Database secrets belum lengkap. Cek Streamlit Secrets.")
        st.exception(e)
        st.stop()


@st.cache_data(ttl=3600, show_spinner="Loading norm database...")
def load_norm_data():
    engine = get_engine()

    query = """
        SELECT
            slice_type,
            study,
            category,
            sub_category,
            detail_product,
            gender,
            age_group,
            actual_age,
            ses,
            occupation,
            type_of_study,
            test_type,
            methodology,
            sub_method,
            num_of_product,
            sequence,
            parameter_id,
            parameter_name,
            parameter_key,
            parameter_group,
            scale,
            norm_grade,
            mean_score,
            tb_pct,
            t2b_pct,
            t3b_pct,
            base,
            min_score,
            max_score,
            std_score
        FROM norm_value_all
    """

    try:
        df = pd.read_sql(query, engine)

    except Exception:
        # fallback untuk tabel lama yang belum punya kolom study / parameter_id / parameter_group
        fallback_query = """
            SELECT *
            FROM norm_value_all
        """
        df = pd.read_sql(fallback_query, engine)

    return df


# ============================================================
# 6. LOAD DATA
# ============================================================

df = load_norm_data()

if df.empty:
    st.error("Tabel norm_value_all kosong. Upload ulang data dari Colab ke Supabase.")
    st.stop()


# ============================================================
# 7. DATA CLEANING FOR APP
# ============================================================

expected_cols = [
    "slice_type",
    "study",
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
    "parameter_id",
    "parameter_name",
    "parameter_key",
    "parameter_group",
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

for col in expected_cols:
    if col not in df.columns:
        df[col] = pd.NA

string_cols = [
    "slice_type",
    "study",
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
    "sequence",
    "parameter_name",
    "parameter_key",
    "parameter_group",
    "norm_grade",
]

for col in string_cols:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace(["", "nan", "NaN", "None", "NULL", "null", "<NA>"], pd.NA)
    )

numeric_cols = [
    "actual_age",
    "parameter_id",
    "scale",
    "mean_score",
    "tb_pct",
    "t2b_pct",
    "t3b_pct",
    "base",
    "min_score",
    "max_score",
    "std_score",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Standard ordering
grade_order = ["Top 25%", "Average 50%", "Bottom 25%"]
df["norm_grade"] = pd.Categorical(
    df["norm_grade"],
    categories=grade_order,
    ordered=True
)


# ============================================================
# 8. SIDEBAR
# ============================================================

logo_b64 = image_to_base64(LOGO_PATH)

with st.sidebar:
    if logo_b64:
        st.markdown(
            f"""
            <div style="text-align:center; margin-bottom:18px;">
                <img src="data:image/png;base64,{logo_b64}" style="max-width:150px;">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown("## DEKA")

    st.markdown("### Filters")

    # Benchmark view
    available_slices = (
        df["slice_type"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    # Put important slices first
    preferred_order = [
        "Global",
        "study",
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
        "sequence",
        "study | category",
        "study | gender",
        "study | age_group",
        "study | ses",
        "study | methodology",
        "category | gender",
        "category | age_group",
        "category | ses",
        "category | methodology",
    ]

    ordered_slices = [
        s for s in preferred_order
        if s in available_slices
    ] + [
        s for s in available_slices
        if s not in preferred_order
    ]

    slice_display_map = {
        readable_slice_label(s): s
        for s in ordered_slices
    }

    selected_slice_label = st.selectbox(
        "Benchmark View",
        options=list(slice_display_map.keys()),
        index=0
    )

    selected_slice = slice_display_map[selected_slice_label]

    filtered = df[df["slice_type"].astype(str).eq(selected_slice)].copy()

    # Dynamic segment filters based on slice_type
    active_dimensions = slice_dimensions(selected_slice)

    selected_segment_filters = {}

    for dim in active_dimensions:
        if dim in filtered.columns:
            options = filter_options(filtered, dim)

            if options:
                selected = st.multiselect(
                    display_col_name(dim),
                    options=options,
                    default=[],
                    placeholder=f"All {display_col_name(dim)}"
                )
                selected_segment_filters[dim] = selected
                filtered = apply_multiselect_filter(filtered, dim, selected)

    st.markdown("---")

    # Extra optional filters
    st.markdown("#### Additional Filters")

    parameter_group_options = filter_options(filtered, "parameter_group")
    selected_parameter_groups = st.multiselect(
        "Parameter Group",
        options=parameter_group_options,
        default=[],
        placeholder="All parameter groups"
    )
    filtered = apply_multiselect_filter(filtered, "parameter_group", selected_parameter_groups)

    parameter_options = filter_options(filtered, "parameter_name")
    selected_parameters = st.multiselect(
        "Parameter",
        options=parameter_options,
        default=[],
        placeholder="All parameters"
    )
    filtered = apply_multiselect_filter(filtered, "parameter_name", selected_parameters)

    scale_options = (
        filtered["scale"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_scales = st.multiselect(
        "Scale",
        options=scale_options,
        default=scale_options,
        placeholder="All scales"
    )

    if selected_scales:
        filtered = filtered[filtered["scale"].isin(selected_scales)].copy()

    norm_options = [
        g for g in grade_order
        if g in filtered["norm_grade"].astype(str).unique().tolist()
    ]

    selected_norm_groups = st.multiselect(
        "Norm Group",
        options=norm_options,
        default=norm_options,
        placeholder="All norm groups"
    )

    if selected_norm_groups:
        filtered = filtered[filtered["norm_grade"].astype(str).isin(selected_norm_groups)].copy()

    min_base = st.number_input(
        "Minimum Base",
        min_value=0,
        value=10,
        step=10
    )

    filtered = filtered[filtered["base"].fillna(0) >= min_base].copy()

    focus_metric_label = st.selectbox(
        "Focus Metric",
        options=["Mean Score", "Top Box", "Top 2 Boxes", "Top 3 Boxes"],
        index=0
    )

    focus_metric = metric_column_from_label(focus_metric_label)

    top_n = st.slider(
        "Top N Parameters",
        min_value=5,
        max_value=50,
        value=15,
        step=5
    )

    st.markdown("---")

    if st.button("Clear cache / refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# 9. HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-kicker">DEKA INSIGHT NORM DATABASE</div>
        <div class="hero-title">Score context, not just score tracking.</div>
        <div class="hero-subtitle">
            Explore historical product-test norms across study, category, product detail,
            demographics, methodology, and attribute groups. Use this dashboard to see
            what is strong, average, or weak against benchmark performance.
        </div>
        <span class="hero-pill">Benchmark: {selected_slice_label}</span>
        <span class="hero-pill">Metric: {focus_metric_label}</span>
        <span class="hero-pill">Rows: {safe_int(len(filtered))}</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. EMPTY STATE
# ============================================================

if filtered.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Coba turunkan Minimum Base atau kosongkan beberapa filter.")
    st.stop()


# ============================================================
# 11. KPI SNAPSHOT
# ============================================================

st.markdown('<div class="section-title">Dashboard Snapshot</div>', unsafe_allow_html=True)

total_rows = len(filtered)
unique_params = filtered["parameter_name"].nunique()
unique_groups = filtered["parameter_group"].nunique()
median_base = filtered["base"].median()
avg_mean = filtered["mean_score"].mean()
avg_tb = filtered["tb_pct"].mean()
avg_t2b = filtered["t2b_pct"].mean()

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Norm Rows</div>
            <div class="metric-value">{safe_int(total_rows)}</div>
            <div class="metric-note">Rows after filters</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Parameters</div>
            <div class="metric-value">{safe_int(unique_params)}</div>
            <div class="metric-note">Unique attributes</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Groups</div>
            <div class="metric-value">{safe_int(unique_groups)}</div>
            <div class="metric-note">Parameter groups</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Median Base</div>
            <div class="metric-value">{safe_int(median_base)}</div>
            <div class="metric-note">{confidence_label(median_base)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c5:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg Mean</div>
            <div class="metric-value">{safe_number(avg_mean, 2)}</div>
            <div class="metric-note">Average score</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Avg T2B</div>
            <div class="metric-value">{safe_number(avg_t2b, 1)}%</div>
            <div class="metric-note">Top 2 Boxes</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 12. FILTER SUMMARY
# ============================================================

with st.expander("Current filter summary", expanded=False):
    st.write("**Benchmark View:**", selected_slice_label)

    if active_dimensions:
        for dim in active_dimensions:
            selected = selected_segment_filters.get(dim, [])
            st.write(f"**{display_col_name(dim)}:**", ", ".join(selected) if selected else "All")

    st.write("**Parameter Group:**", ", ".join(selected_parameter_groups) if selected_parameter_groups else "All")
    st.write("**Parameter:**", ", ".join(selected_parameters) if selected_parameters else "All")
    st.write("**Scale:**", ", ".join(map(str, selected_scales)) if selected_scales else "All")
    st.write("**Norm Group:**", ", ".join(selected_norm_groups) if selected_norm_groups else "All")
    st.write("**Minimum Base:**", min_base)


# ============================================================
# 13. KEY INSIGHTS
# ============================================================

st.markdown('<div class="section-title">Key Signals</div>', unsafe_allow_html=True)

metric_df = filtered.dropna(subset=[focus_metric]).copy()

if not metric_df.empty:
    best_row = metric_df.sort_values(focus_metric, ascending=False).iloc[0]
    low_row = metric_df.sort_values(focus_metric, ascending=True).iloc[0]
    strongest_base_row = metric_df.sort_values("base", ascending=False).iloc[0]

    i1, i2, i3 = st.columns(3)

    suffix = metric_suffix(focus_metric)

    with i1:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Strongest Signal</div>
                <div class="insight-main">{clean_display_value(best_row["parameter_name"])}</div>
                <div class="insight-sub">
                    {metric_display_name(focus_metric)}: <b>{safe_number(best_row[focus_metric], 2)}{suffix}</b><br>
                    Norm: {clean_display_value(best_row["norm_grade"])}<br>
                    Base: {safe_int(best_row["base"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i2:
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Watch-out Area</div>
                <div class="insight-main">{clean_display_value(low_row["parameter_name"])}</div>
                <div class="insight-sub">
                    {metric_display_name(focus_metric)}: <b>{safe_number(low_row[focus_metric], 2)}{suffix}</b><br>
                    Norm: {clean_display_value(low_row["norm_grade"])}<br>
                    Base: {safe_int(low_row["base"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with i3:
        conf_class = confidence_class(strongest_base_row["base"])
        st.markdown(
            f"""
            <div class="insight-card">
                <div class="insight-title">Largest Evidence Base</div>
                <div class="insight-main">{clean_display_value(strongest_base_row["parameter_name"])}</div>
                <div class="insight-sub">
                    Base: <b>{safe_int(strongest_base_row["base"])}</b><br>
                    Mean: {safe_number(strongest_base_row["mean_score"], 2)}<br>
                    Norm: {clean_display_value(strongest_base_row["norm_grade"])}
                </div>
                <span class="badge badge-{conf_class}">{confidence_label(strongest_base_row["base"])}</span>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 14. CHARTS
# ============================================================

st.markdown('<div class="section-title">Benchmark Ranking</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Top parameters based on selected focus metric.</div>',
    unsafe_allow_html=True
)

chart_df = (
    metric_df
    .groupby(["parameter_name", "parameter_group", "scale"], dropna=False)
    .agg(
        metric_value=(focus_metric, "mean"),
        base=("base", "sum"),
        mean_score=("mean_score", "mean"),
        tb_pct=("tb_pct", "mean"),
        t2b_pct=("t2b_pct", "mean"),
        t3b_pct=("t3b_pct", "mean"),
    )
    .reset_index()
    .sort_values("metric_value", ascending=False)
    .head(top_n)
)

if not chart_df.empty:
    fig = px.bar(
        chart_df.sort_values("metric_value", ascending=True),
        x="metric_value",
        y="parameter_name",
        orientation="h",
        color="parameter_group",
        hover_data=["scale", "base", "mean_score", "tb_pct", "t2b_pct", "t3b_pct"],
        labels={
            "metric_value": metric_display_name(focus_metric),
            "parameter_name": "Parameter",
            "parameter_group": "Parameter Group",
        },
        height=max(450, top_n * 28)
    )

    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
        legend_title_text="Parameter Group",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Data chart tidak tersedia untuk filter ini.")


# ============================================================
# 15. NORM GRADE COMPARISON
# ============================================================

st.markdown('<div class="section-title">Norm Group Comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Compare Top 25%, Average 50%, and Bottom 25% across selected data.</div>',
    unsafe_allow_html=True
)

grade_chart_df = (
    filtered
    .dropna(subset=["norm_grade", focus_metric])
    .groupby(["norm_grade"], observed=False)
    .agg(
        metric_value=(focus_metric, "mean"),
        base=("base", "sum"),
        rows=("parameter_name", "count")
    )
    .reset_index()
)

if not grade_chart_df.empty:
    grade_chart_df["norm_grade"] = grade_chart_df["norm_grade"].astype(str)

    fig2 = px.bar(
        grade_chart_df,
        x="norm_grade",
        y="metric_value",
        hover_data=["base", "rows"],
        labels={
            "norm_grade": "Norm Group",
            "metric_value": metric_display_name(focus_metric)
        },
        height=380
    )

    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10)
    )

    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 16. SCORE CHECKER
# ============================================================

st.markdown('<div class="section-title">Score Checker</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Input a product score and compare it against the selected benchmark norms.</div>',
    unsafe_allow_html=True
)

checker_df = filtered.dropna(subset=["parameter_name", "scale", "norm_grade", "mean_score"]).copy()

if not checker_df.empty:
    checker_df["param_scale_label"] = (
        checker_df["parameter_name"].astype(str)
        + " | Scale "
        + checker_df["scale"].astype("Int64").astype(str)
    )

    checker_options = (
        checker_df["param_scale_label"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    cc1, cc2, cc3 = st.columns([2, 1, 1])

    with cc1:
        selected_param_scale = st.selectbox(
            "Choose Parameter",
            options=checker_options
        )

    temp_checker = checker_df[checker_df["param_scale_label"].eq(selected_param_scale)].copy()

    selected_scale = int(temp_checker["scale"].dropna().iloc[0])

    with cc2:
        input_score = st.number_input(
            "Input Score",
            min_value=0.0,
            max_value=float(selected_scale),
            value=float(min(selected_scale, max(1, round(temp_checker["mean_score"].mean(), 1)))),
            step=0.1
        )

    with cc3:
        st.metric("Scale", selected_scale)

    grade_ref = (
        temp_checker
        .groupby("norm_grade", observed=False)
        .agg(mean_score=("mean_score", "mean"), base=("base", "sum"))
        .reset_index()
    )

    grade_ref["distance"] = abs(grade_ref["mean_score"] - input_score)
    nearest = grade_ref.sort_values("distance").iloc[0]

    st.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Closest Benchmark Tier</div>
            <div class="insight-main">{clean_display_value(nearest["norm_grade"])}</div>
            <div class="insight-sub">
                Your score: <b>{safe_number(input_score, 2)}</b><br>
                Closest norm mean: <b>{safe_number(nearest["mean_score"], 2)}</b><br>
                Norm base: <b>{safe_int(nearest["base"])}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.dataframe(
        grade_ref[["norm_grade", "mean_score", "base"]].rename(columns=display_col_name),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 17. STUDY / PROJECT QUICK VIEW
# ============================================================

if "study" in df.columns and df["study"].notna().any():
    st.markdown('<div class="section-title">Study / Project Coverage</div>', unsafe_allow_html=True)

    study_df = (
        df[df["slice_type"].astype(str).eq("study")]
        .dropna(subset=["study"])
        .groupby("study")
        .agg(
            rows=("parameter_name", "count"),
            parameters=("parameter_name", "nunique"),
            median_base=("base", "median"),
            avg_mean=("mean_score", "mean"),
            avg_t2b=("t2b_pct", "mean")
        )
        .reset_index()
        .sort_values("study")
    )

    if not study_df.empty:
        fig3 = px.bar(
            study_df,
            x="study",
            y="rows",
            hover_data=["parameters", "median_base", "avg_mean", "avg_t2b"],
            labels={
                "study": "Study / Project",
                "rows": "Norm Rows"
            },
            height=420
        )

        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-35,
            margin=dict(l=10, r=10, t=30, b=80)
        )

        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Slice 'study' belum tersedia di norm_value_all. Upload ulang data dengan slice study.")


# ============================================================
# 18. DATA TABLE
# ============================================================

st.markdown('<div class="section-title">Norm Table</div>', unsafe_allow_html=True)

visible_segment_cols = [
    "study",
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

base_cols = [
    "slice_type",
    "parameter_id",
    "parameter_name",
    "parameter_group",
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

table_cols = [
    col for col in visible_segment_cols + base_cols
    if col in filtered.columns
]

table_df = filtered[table_cols].copy()

for col in table_df.columns:
    if str(table_df[col].dtype) in ["object", "string", "category"]:
        table_df[col] = table_df[col].astype(str).replace(["nan", "None", "<NA>"], "—")

table_df = table_df.rename(columns=display_col_name)

st.dataframe(
    table_df.head(500),
    use_container_width=True,
    hide_index=True
)

st.caption(f"Showing first 500 rows out of {len(table_df):,} filtered rows.")

st.download_button(
    label="Download filtered table as CSV",
    data=make_download_csv(table_df),
    file_name="deka_norm_filtered_table.csv",
    mime="text/csv"
)


# ============================================================
# 19. RAW DEBUG / DATABASE CHECK
# ============================================================

with st.expander("Database check", expanded=False):
    st.write("Rows loaded:", len(df))
    st.write("Columns loaded:", df.columns.tolist())

    st.write("Available benchmark views:")
    st.dataframe(
        df["slice_type"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "slice_type", "slice_type": "rows"}),
        use_container_width=True,
        hide_index=True
    )

    if "study" in df.columns:
        st.write("Available studies:")
        st.dataframe(
            df["study"]
            .dropna()
            .astype(str)
            .value_counts()
            .reset_index()
            .rename(columns={"index": "study", "study": "rows"}),
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# 20. FOOTER
# ============================================================

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="small-note">
        <b>Reading guide:</b> Top 25% represents the stronger-performing benchmark tier,
        Average 50% represents the middle benchmark range, and Bottom 25% represents the weaker-performing tier.
        Use base size as a confidence signal before making decisions.
    </div>
    """,
    unsafe_allow_html=True
)
