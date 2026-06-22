# ============================================================
# DEKA INSIGHT — NORM DASHBOARD
# app.py
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
# 1. ASSETS & PAGE CONFIG
# ============================================================

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "deka-logo.png"
ICON_PATH = BASE_DIR / "deka-icon.png"


def image_to_base64(path):
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


logo_b64 = image_to_base64(LOGO_PATH)

if ICON_PATH.exists():
    page_icon = str(ICON_PATH)
else:
    page_icon = "📊"

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon=page_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. HELPERS
# ============================================================

def safe_num(x, decimals=1):
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


def clean_value(x):
    if pd.isna(x):
        return "—"
    x = str(x).strip()
    if x.lower() in ["", "nan", "none", "null", "<na>"]:
        return "—"
    return x


def get_options(df, col):
    if col not in df.columns:
        return []

    return (
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
        .replace(["", "nan", "None", "NULL", "<NA>", "null"], np.nan)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


def filter_df(df, col, selected):
    if col not in df.columns or not selected:
        return df
    return df[df[col].astype(str).isin(selected)].copy()


def metric_col(label):
    return {
        "Mean Score": "mean_score",
        "Top Box": "tb_pct",
        "Top 2 Boxes": "t2b_pct",
        "Top 3 Boxes": "t3b_pct",
    }.get(label, "mean_score")


def metric_label(col):
    return {
        "mean_score": "Mean Score",
        "tb_pct": "Top Box",
        "t2b_pct": "Top 2 Boxes",
        "t3b_pct": "Top 3 Boxes",
    }.get(col, col)


def metric_suffix(col):
    return "%" if col in ["tb_pct", "t2b_pct", "t3b_pct"] else ""


def confidence(base):
    if pd.isna(base):
        return "No base"
    if base >= 500:
        return "Strong base"
    if base >= 100:
        return "Reliable base"
    if base >= 30:
        return "Directional"
    return "Low base"


def dims_from_slice(slice_type):
    if slice_type == "Global":
        return []
    return [x.strip() for x in str(slice_type).split("|")]


def slice_label(slice_type):
    mapping = {
        "Global": "Overall",
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
    return mapping.get(slice_type, str(slice_type).replace("_", " ").title())


def col_label(col):
    mapping = {
        "slice_type": "View",
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
        "num_of_product": "# Product",
        "sequence": "Sequence",
        "parameter_name": "Parameter",
        "parameter_group": "Group",
        "scale": "Scale",
        "norm_grade": "Norm",
        "mean_score": "Mean",
        "tb_pct": "TB",
        "t2b_pct": "T2B",
        "t3b_pct": "T3B",
        "base": "Base",
    }
    return mapping.get(col, col.replace("_", " ").title())


# ============================================================
# 3. CSS
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #090F25;
        --navy2: #18244A;
        --gold: #F2A93B;
        --cream: #FAF7F0;
        --card: #FFFFFF;
        --line: #E8E1D8;
        --text: #0B1026;
        --muted: #737B8E;
        --green: #2E7D5B;
        --red: #C9453F;
    }

    .stApp {
        background: linear-gradient(180deg, #FAF7F0 0%, #FFFFFF 43%, #FAF7F0 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 3.8rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }

    section[data-testid="stSidebar"] {
        background: var(--navy);
        border-right: 1px solid rgba(255,255,255,.06);
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #F7F8FF !important;
    }

    section[data-testid="stSidebar"] .stSelectbox div,
    section[data-testid="stSidebar"] .stMultiSelect div,
    section[data-testid="stSidebar"] .stNumberInput div {
        color: #0B1026 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {
        background-color: var(--gold) !important;
        border-radius: 999px !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] span {
        color: #0B1026 !important;
        font-weight: 750 !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {
        fill: #0B1026 !important;
    }

    .sidebar-logo {
        text-align: center;
        margin: 8px 0 18px 0;
    }

    .sidebar-logo img {
        max-width: 136px;
    }

    .filter-hint {
        font-size: 12px;
        color: rgba(255,255,255,.60);
        line-height: 1.35;
        margin: -2px 0 14px 0;
    }

    .hero {
        background: linear-gradient(135deg, #111A39 0%, #192653 100%);
        border-radius: 22px;
        padding: 22px 30px;
        color: white;
        box-shadow: 0 16px 38px rgba(9,15,37,.16);
        margin-top: 10px;
        margin-bottom: 18px;
    }

    .kicker {
        color: var(--gold);
        font-size: 10.5px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 900;
        line-height: 1.08;
        margin: 0 0 7px 0;
    }

    .hero-sub {
        color: rgba(255,255,255,.74);
        font-size: 14px;
        line-height: 1.45;
        max-width: 720px;
    }

    .pill {
        display: inline-block;
        margin-top: 12px;
        margin-right: 7px;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(242,169,59,.12);
        border: 1px solid rgba(242,169,59,.30);
        color: var(--gold);
        font-size: 11px;
        font-weight: 850;
    }

    .section-title {
        font-size: 20px;
        font-weight: 900;
        color: var(--text);
        margin: 19px 0 6px 0;
    }

    .section-sub {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 12px;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 16px 17px;
        box-shadow: 0 10px 26px rgba(11,16,38,.052);
        height: 100%;
    }

    .metric-label {
        color: var(--muted);
        font-size: 10.5px;
        font-weight: 900;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--text);
        font-size: 27px;
        font-weight: 900;
        line-height: 1.1;
    }

    .metric-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 7px;
    }

    .insight-title {
        font-size: 12px;
        font-weight: 900;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: 8px;
    }

    .insight-main {
        font-size: 21px;
        font-weight: 900;
        color: var(--text);
        line-height: 1.18;
        margin-bottom: 9px;
    }

    .insight-sub {
        color: var(--muted);
        font-size: 12.5px;
        line-height: 1.48;
    }

    .badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 850;
        margin-top: 8px;
    }

    .badge-green {
        background: #E8F5EE;
        color: #2E7D5B;
    }

    .badge-gold {
        background: #FFF3D8;
        color: #A76500;
    }

    .badge-red {
        background: #FDECEC;
        color: #C9453F;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--line);
    }

    .stDownloadButton > button {
        background: var(--navy) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 999px !important;
        font-weight: 850 !important;
        padding: .55rem 1.2rem !important;
    }

    .stButton > button {
        border-radius: 999px !important;
        font-weight: 850 !important;
    }

    hr {
        margin-top: 1.4rem;
        margin-bottom: 1.1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 4. DATABASE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine():
    try:
        user = st.secrets["postgres"]["user"]
        password = quote_plus(st.secrets["postgres"]["password"])
        host = st.secrets["postgres"]["host"]
        port = st.secrets["postgres"]["port"]
        database = st.secrets["postgres"]["database"]

        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"
        return create_engine(url)

    except Exception as e:
        st.error("Database secrets belum lengkap. Cek Settings → Secrets di Streamlit Cloud.")
        st.exception(e)
        st.stop()


@st.cache_data(ttl=3600, show_spinner="Loading norm database...")
def load_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM norm_value_all", engine)


df = load_data()

if df.empty:
    st.error("Tabel norm_value_all kosong. Upload ulang data dari Colab.")
    st.stop()


# ============================================================
# 5. CLEAN DATA
# ============================================================

expected_cols = [
    "slice_type", "study", "category", "sub_category", "detail_product",
    "gender", "age_group", "actual_age", "ses", "occupation",
    "type_of_study", "test_type", "methodology", "sub_method",
    "num_of_product", "sequence", "parameter_id", "parameter_name",
    "parameter_key", "parameter_group", "scale", "norm_grade",
    "mean_score", "tb_pct", "t2b_pct", "t3b_pct", "base",
    "min_score", "max_score", "std_score",
]

for col in expected_cols:
    if col not in df.columns:
        df[col] = pd.NA

text_cols = [
    "slice_type", "study", "category", "sub_category", "detail_product",
    "gender", "age_group", "ses", "occupation", "type_of_study",
    "test_type", "methodology", "sub_method", "sequence",
    "parameter_name", "parameter_key", "parameter_group", "norm_grade",
]

for col in text_cols:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace(["", "nan", "NaN", "None", "none", "NULL", "null", "<NA>"], pd.NA)
    )

num_cols = [
    "actual_age", "num_of_product", "parameter_id", "scale",
    "mean_score", "tb_pct", "t2b_pct", "t3b_pct", "base",
    "min_score", "max_score", "std_score",
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

grade_order = ["Top 25%", "Average 50%", "Bottom 25%"]
df["norm_grade"] = pd.Categorical(df["norm_grade"], categories=grade_order, ordered=True)


# ============================================================
# 6. SIDEBAR FILTERS
# ============================================================

with st.sidebar:
    if logo_b64:
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{logo_b64}">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("## deka insight")

    st.markdown("## Filters")
    st.markdown(
        '<div class="filter-hint">Empty filter means all values are included.</div>',
        unsafe_allow_html=True,
    )

    available_slices = (
        df["slice_type"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    preferred = [
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
        "methodology | test_type",
    ]

    ordered_slices = [x for x in preferred if x in available_slices]
    ordered_slices += sorted([x for x in available_slices if x not in ordered_slices])

    slice_display = {slice_label(x): x for x in ordered_slices}

    default_label = "Study / Project" if "study" in available_slices else slice_label(ordered_slices[0])

    selected_slice_label = st.selectbox(
        "Benchmark view",
        options=list(slice_display.keys()),
        index=list(slice_display.keys()).index(default_label) if default_label in slice_display else 0,
    )

    selected_slice = slice_display[selected_slice_label]
    work = df[df["slice_type"].astype(str).eq(selected_slice)].copy()

    active_dims = dims_from_slice(selected_slice)
    selected_dim_filters = {}

    for dim in active_dims:
        options = get_options(work, dim)

        if options:
            selected = st.multiselect(
                col_label(dim),
                options=options,
                default=[],
                placeholder=f"All {col_label(dim)}",
            )

            selected_dim_filters[dim] = selected
            work = filter_df(work, dim, selected)

    st.markdown("---")
    st.markdown("### Refine")

    selected_metric_label = st.selectbox(
        "Metric",
        options=["Mean Score", "Top Box", "Top 2 Boxes", "Top 3 Boxes"],
        index=0,
    )

    selected_metric = metric_col(selected_metric_label)

    group_options = get_options(work, "parameter_group")
    selected_groups = st.multiselect(
        "Parameter group",
        options=group_options,
        default=[],
        placeholder="All groups",
    )
    work = filter_df(work, "parameter_group", selected_groups)

    parameter_options = get_options(work, "parameter_name")
    selected_params = st.multiselect(
        "Parameter",
        options=parameter_options,
        default=[],
        placeholder="All parameters",
    )
    work = filter_df(work, "parameter_name", selected_params)

    scale_options = (
        work["scale"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    selected_scales = st.multiselect(
        "Scale",
        options=scale_options,
        default=[],
        placeholder="All scales",
    )

    if selected_scales:
        work = work[work["scale"].isin(selected_scales)].copy()

    norm_options = [x for x in grade_order if x in work["norm_grade"].astype(str).unique().tolist()]

    selected_norms = st.multiselect(
        "Norm group",
        options=norm_options,
        default=[],
        placeholder="All norm groups",
    )

    if selected_norms:
        work = work[work["norm_grade"].astype(str).isin(selected_norms)].copy()

    min_base = st.number_input("Minimum base", min_value=0, value=10, step=10)
    work = work[work["base"].fillna(0) >= min_base].copy()

    top_n = st.slider("Top attributes", min_value=5, max_value=20, value=10, step=1)

    st.markdown("---")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# 7. HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="kicker">DEKA INSIGHT NORM DATABASE</div>
        <div class="hero-title">Product norm benchmark.</div>
        <div class="hero-sub">
            Benchmark product-test performance by study, segment, method, and attribute.
        </div>
        <span class="pill">{selected_slice_label}</span>
        <span class="pill">{selected_metric_label}</span>
        <span class="pill">{safe_int(len(work))} rows</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if work.empty:
    st.warning("No data for the selected filters. Remove a filter or lower the minimum base.")
    st.stop()


# ============================================================
# 8. KPI SNAPSHOT
# ============================================================

metric_df = work.dropna(subset=[selected_metric]).copy()
suffix = metric_suffix(selected_metric)

top_tier_mean = np.nan
bottom_tier_mean = np.nan
tier_gap = np.nan

tier_summary_for_kpi = (
    metric_df
    .dropna(subset=["norm_grade"])
    .groupby("norm_grade", observed=False)
    .agg(value=(selected_metric, "mean"))
    .reset_index()
)

if "Top 25%" in tier_summary_for_kpi["norm_grade"].astype(str).tolist():
    top_tier_mean = tier_summary_for_kpi.loc[
        tier_summary_for_kpi["norm_grade"].astype(str).eq("Top 25%"),
        "value"
    ].iloc[0]

if "Bottom 25%" in tier_summary_for_kpi["norm_grade"].astype(str).tolist():
    bottom_tier_mean = tier_summary_for_kpi.loc[
        tier_summary_for_kpi["norm_grade"].astype(str).eq("Bottom 25%"),
        "value"
    ].iloc[0]

if not pd.isna(top_tier_mean) and not pd.isna(bottom_tier_mean):
    tier_gap = top_tier_mean - bottom_tier_mean

study_count = work["study"].dropna().nunique()
attribute_count = work["parameter_name"].dropna().nunique()
median_base = work["base"].median()
avg_metric = metric_df[selected_metric].mean()

st.markdown('<div class="section-title">Performance Snapshot</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

kpis = [
    ("Studies", safe_int(study_count), "Projects covered"),
    ("Attributes", safe_int(attribute_count), "Parameters compared"),
    ("Median Base", safe_int(median_base), confidence(median_base)),
    (f"Avg {selected_metric_label}", f"{safe_num(avg_metric, 2)}{suffix}", "Filtered benchmark"),
    ("Tier Gap", f"{safe_num(tier_gap, 2)}{suffix}", "Top 25% minus Bottom 25%"),
]

for col, (label, value, note) in zip([k1, k2, k3, k4, k5], kpis):
    with col:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 9. KEY INSIGHTS
# ============================================================

st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

if not metric_df.empty:
    attr_summary = (
        metric_df
        .groupby(["parameter_name", "parameter_group"], dropna=False)
        .agg(
            value=(selected_metric, "mean"),
            base=("base", "sum"),
            mean_score=("mean_score", "mean"),
        )
        .reset_index()
    )

    best_attr = attr_summary.sort_values("value", ascending=False).iloc[0]
    weak_attr = attr_summary.sort_values("value", ascending=True).iloc[0]

    if "study" in metric_df.columns and metric_df["study"].notna().any():
        study_summary = (
            metric_df
            .dropna(subset=["study"])
            .groupby("study")
            .agg(
                value=(selected_metric, "mean"),
                base=("base", "sum"),
                attributes=("parameter_name", "nunique"),
            )
            .reset_index()
        )

        best_study = study_summary.sort_values("value", ascending=False).iloc[0]
        weak_study = study_summary.sort_values("value", ascending=True).iloc[0]
    else:
        best_study = None
        weak_study = None

    i1, i2, i3, i4 = st.columns(4)

    with i1:
        st.markdown(
            f"""
            <div class="card">
                <div class="insight-title">Best Attribute</div>
                <div class="insight-main">{clean_value(best_attr["parameter_name"])}</div>
                <div class="insight-sub">
                    {selected_metric_label}: <b>{safe_num(best_attr["value"], 2)}{suffix}</b><br>
                    Group: {clean_value(best_attr["parameter_group"])}<br>
                    Base: {safe_int(best_attr["base"])}
                </div>
                <span class="badge badge-green">Strength</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i2:
        st.markdown(
            f"""
            <div class="card">
                <div class="insight-title">Watch-out Attribute</div>
                <div class="insight-main">{clean_value(weak_attr["parameter_name"])}</div>
                <div class="insight-sub">
                    {selected_metric_label}: <b>{safe_num(weak_attr["value"], 2)}{suffix}</b><br>
                    Group: {clean_value(weak_attr["parameter_group"])}<br>
                    Base: {safe_int(weak_attr["base"])}
                </div>
                <span class="badge badge-red">Review</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with i3:
        if best_study is not None:
            st.markdown(
                f"""
                <div class="card">
                    <div class="insight-title">Leading Study</div>
                    <div class="insight-main">{clean_value(best_study["study"])}</div>
                    <div class="insight-sub">
                        {selected_metric_label}: <b>{safe_num(best_study["value"], 2)}{suffix}</b><br>
                        Attributes: {safe_int(best_study["attributes"])}<br>
                        Base: {safe_int(best_study["base"])}
                    </div>
                    <span class="badge badge-green">Leading</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="insight-title">Leading Study</div>
                    <div class="insight-main">—</div>
                    <div class="insight-sub">Study is not available in this view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with i4:
        if weak_study is not None:
            st.markdown(
                f"""
                <div class="card">
                    <div class="insight-title">Lowest Study</div>
                    <div class="insight-main">{clean_value(weak_study["study"])}</div>
                    <div class="insight-sub">
                        {selected_metric_label}: <b>{safe_num(weak_study["value"], 2)}{suffix}</b><br>
                        Attributes: {safe_int(weak_study["attributes"])}<br>
                        Base: {safe_int(weak_study["base"])}
                    </div>
                    <span class="badge badge-gold">Compare</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="card">
                    <div class="insight-title">Tier Gap</div>
                    <div class="insight-main">{safe_num(tier_gap, 2)}{suffix}</div>
                    <div class="insight-sub">Top 25% minus Bottom 25%.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# 10. MAIN CHARTS
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="section-title">Attribute Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Highest-performing attributes for the selected benchmark.</div>', unsafe_allow_html=True)

    rank_df = (
        metric_df
        .groupby(["parameter_name", "parameter_group"], dropna=False)
        .agg(
            value=(selected_metric, "mean"),
            base=("base", "sum"),
        )
        .reset_index()
        .sort_values("value", ascending=False)
        .head(top_n)
    )

    if not rank_df.empty:
        fig_rank = px.bar(
            rank_df.sort_values("value", ascending=True),
            x="value",
            y="parameter_name",
            orientation="h",
            color="parameter_group",
            hover_data=["base"],
            labels={
                "value": selected_metric_label,
                "parameter_name": "",
                "parameter_group": "Group",
            },
            height=max(350, top_n * 32),
        )

        fig_rank.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=15, b=5),
            legend_title_text="Group",
        )

        st.plotly_chart(fig_rank, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Norm Tier Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average score by benchmark tier.</div>', unsafe_allow_html=True)

    tier_df = (
        metric_df
        .dropna(subset=["norm_grade"])
        .groupby("norm_grade", observed=False)
        .agg(
            value=(selected_metric, "mean"),
            base=("base", "sum"),
            rows=("parameter_name", "count"),
        )
        .reset_index()
    )

    if not tier_df.empty:
        tier_df["norm_grade"] = tier_df["norm_grade"].astype(str)

        fig_tier = px.bar(
            tier_df,
            x="norm_grade",
            y="value",
            hover_data=["base", "rows"],
            labels={
                "norm_grade": "",
                "value": selected_metric_label,
            },
            height=350,
        )

        fig_tier.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=15, b=5),
            showlegend=False,
        )

        st.plotly_chart(fig_tier, use_container_width=True)


# ============================================================
# 11. ATTRIBUTE GAP & STUDY PERFORMANCE
# ============================================================

left2, right2 = st.columns([1.2, 1])

with left2:
    st.markdown('<div class="section-title">Attribute Discrimination</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Gap between Top 25% and Bottom 25% by attribute.</div>', unsafe_allow_html=True)

    gap_df = (
        metric_df
        .dropna(subset=["norm_grade"])
        .pivot_table(
            index=["parameter_name", "parameter_group"],
            columns="norm_grade",
            values=selected_metric,
            aggfunc="mean",
        )
        .reset_index()
    )

    if {"Top 25%", "Bottom 25%"}.issubset(gap_df.columns):
        gap_df["gap"] = gap_df["Top 25%"] - gap_df["Bottom 25%"]
        gap_df = gap_df.dropna(subset=["gap"]).sort_values("gap", ascending=False).head(top_n)

        if not gap_df.empty:
            fig_gap = px.bar(
                gap_df.sort_values("gap", ascending=True),
                x="gap",
                y="parameter_name",
                orientation="h",
                color="parameter_group",
                labels={
                    "gap": f"Tier Gap ({selected_metric_label})",
                    "parameter_name": "",
                    "parameter_group": "Group",
                },
                hover_data=["Top 25%", "Bottom 25%"],
                height=max(350, top_n * 32),
            )

            fig_gap.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=15, b=5),
                legend_title_text="Group",
            )

            st.plotly_chart(fig_gap, use_container_width=True)

with right2:
    st.markdown('<div class="section-title">Study Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average benchmark performance by study/project.</div>', unsafe_allow_html=True)

    if "study" in metric_df.columns and metric_df["study"].notna().any():
        study_perf = (
            metric_df
            .dropna(subset=["study"])
            .groupby("study")
            .agg(
                value=(selected_metric, "mean"),
                base=("base", "sum"),
                attributes=("parameter_name", "nunique"),
            )
            .reset_index()
            .sort_values("value", ascending=False)
        )

        if not study_perf.empty:
            fig_study = px.scatter(
                study_perf,
                x="base",
                y="value",
                size="attributes",
                hover_name="study",
                labels={
                    "base": "Total Base",
                    "value": selected_metric_label,
                    "attributes": "Attributes",
                },
                height=350,
            )

            fig_study.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=5, r=5, t=15, b=5),
            )

            st.plotly_chart(fig_study, use_container_width=True)
    else:
        st.info("Study data is not available in the selected view.")


# ============================================================
# 12. NORM TABLE
# ============================================================

st.markdown('<div class="section-title">Norm Table</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Filtered benchmark table for export or detailed review.</div>', unsafe_allow_html=True)

active_segment_cols = active_dims.copy()

base_cols = [
    "parameter_name",
    "parameter_group",
    "scale",
    "norm_grade",
    "mean_score",
    "tb_pct",
    "t2b_pct",
    "t3b_pct",
    "base",
]

table_cols = [c for c in active_segment_cols + base_cols if c in work.columns]

table = work[table_cols].copy()

for col in table.columns:
    if str(table[col].dtype) in ["object", "string", "category"]:
        table[col] = table[col].astype(str).replace(["nan", "None", "<NA>", "null"], "—")

table = table.rename(columns=col_label)

st.dataframe(
    table.head(300),
    use_container_width=True,
    hide_index=True,
    height=330,
)

st.caption(f"Showing 300 of {len(table):,} filtered rows.")

st.download_button(
    "Download CSV",
    data=table.to_csv(index=False).encode("utf-8"),
    file_name="deka_norm_filtered.csv",
    mime="text/csv",
)


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "Top 25% = stronger benchmark tier. Bottom 25% = weaker benchmark tier. Interpret every result together with base size."
)
