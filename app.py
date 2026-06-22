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


PAGE_ICON = str(ICON_PATH) if ICON_PATH.exists() else "📊"

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

logo_b64 = image_to_base64(LOGO_PATH)


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

    values = (
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

    return values


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
        "parameter_id": "ID",
        "parameter_name": "Parameter",
        "parameter_group": "Group",
        "scale": "Scale",
        "norm_grade": "Norm",
        "mean_score": "Mean",
        "tb_pct": "TB",
        "t2b_pct": "T2B",
        "t3b_pct": "T3B",
        "base": "Base",
        "min_score": "Min",
        "max_score": "Max",
        "std_score": "Std",
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
        --navy2: #162044;
        --gold: #F2A93B;
        --cream: #FAF7F0;
        --card: #FFFFFF;
        --line: #E8E1D8;
        --text: #0B1026;
        --muted: #737B8E;
        --green: #2E7D5B;
        --red: #D95F59;
        --soft-gold: #FFF4DF;
        --soft-blue: #F2F5FB;
    }

    .stApp {
        background: linear-gradient(180deg, #FAF7F0 0%, #FFFFFF 42%, #FAF7F0 100%);
        color: var(--text);
    }

    .block-container {
        padding-top: 1.1rem;
        padding-bottom: 2.4rem;
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
        color: var(--text) !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {
        background-color: #F2A93B !important;
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
        max-width: 138px;
    }

    .filter-hint {
        font-size: 12px;
        color: rgba(255,255,255,.60);
        line-height: 1.35;
        margin: -2px 0 14px 0;
    }

    .hero {
        background: linear-gradient(135deg, #111A39 0%, #18244A 72%, #202D58 100%);
        border-radius: 22px;
        padding: 24px 30px;
        color: white;
        box-shadow: 0 16px 38px rgba(9,15,37,.16);
        margin-bottom: 20px;
    }

    .kicker {
        color: var(--gold);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .hero-title {
        font-size: 34px;
        font-weight: 900;
        line-height: 1.08;
        margin: 0 0 7px 0;
    }

    .hero-sub {
        color: rgba(255,255,255,.75);
        font-size: 14px;
        line-height: 1.5;
        max-width: 760px;
    }

    .pill {
        display: inline-block;
        margin-top: 13px;
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
        margin: 20px 0 6px 0;
    }

    .section-sub {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 13px;
    }

    .card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 17px 17px;
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
        font-size: 22px;
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
        margin-top: 1.6rem;
        margin-bottom: 1.2rem;
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
# 6. SIDEBAR
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

    selected_metric_label = st.selectbox(
        "Metric",
        options=["Mean Score", "Top Box", "Top 2 Boxes", "Top 3 Boxes"],
        index=0,
    )

    selected_metric = metric_col(selected_metric_label)

    top_n = st.slider("Top attributes", min_value=5, max_value=25, value=10, step=1)

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
            Read product-test performance by study, segment, method, and attribute.
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
# 8. SNAPSHOT
# ============================================================

st.markdown('<div class="section-title">Snapshot</div>', unsafe_allow_html=True)

total_rows = len(work)
unique_params = work["parameter_name"].nunique()
unique_groups = work["parameter_group"].nunique()
median_base = work["base"].median()
avg_mean = work["mean_score"].mean()
avg_t2b = work["t2b_pct"].mean()

m1, m2, m3, m4, m5 = st.columns(5)

snapshot = [
    ("Rows", safe_int(total_rows), "After filters"),
    ("Attributes", safe_int(unique_params), "Unique parameters"),
    ("Groups", safe_int(unique_groups), "Attribute groups"),
    ("Median Base", safe_int(median_base), confidence(median_base)),
    ("Avg T2B", f"{safe_num(avg_t2b, 1)}%", "Top 2 Boxes"),
]

for col, (label, value, note) in zip([m1, m2, m3, m4, m5], snapshot):
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
# 9. INSIGHTS
# ============================================================

st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

metric_df = work.dropna(subset=[selected_metric]).copy()

if not metric_df.empty:
    suffix = metric_suffix(selected_metric)

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
    else:
        best_study = None

    grade_summary = (
        metric_df
        .dropna(subset=["norm_grade"])
        .groupby("norm_grade", observed=False)
        .agg(value=(selected_metric, "mean"))
        .reset_index()
    )

    spread_text = "—"
    if {"Top 25%", "Bottom 25%"}.issubset(set(grade_summary["norm_grade"].astype(str))):
        top_value = grade_summary.loc[grade_summary["norm_grade"].astype(str).eq("Top 25%"), "value"].iloc[0]
        bottom_value = grade_summary.loc[grade_summary["norm_grade"].astype(str).eq("Bottom 25%"), "value"].iloc[0]
        spread_text = f"{safe_num(top_value - bottom_value, 2)}{suffix}"

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            f"""
            <div class="card">
                <div class="insight-title">Top Attribute</div>
                <div class="insight-main">{clean_value(best_attr["parameter_name"])}</div>
                <div class="insight-sub">
                    {metric_label(selected_metric)}: <b>{safe_num(best_attr["value"], 2)}{suffix}</b><br>
                    Group: {clean_value(best_attr["parameter_group"])}<br>
                    Base: {safe_int(best_attr["base"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"""
            <div class="card">
                <div class="insight-title">Watch-out</div>
                <div class="insight-main">{clean_value(weak_attr["parameter_name"])}</div>
                <div class="insight-sub">
                    {metric_label(selected_metric)}: <b>{safe_num(weak_attr["value"], 2)}{suffix}</b><br>
                    Group: {clean_value(weak_attr["parameter_group"])}<br>
                    Base: {safe_int(weak_attr["base"])}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c3:
        if best_study is not None:
            st.markdown(
                f"""
                <div class="card">
                    <div class="insight-title">Best Study</div>
                    <div class="insight-main">{clean_value(best_study["study"])}</div>
                    <div class="insight-sub">
                        {metric_label(selected_metric)}: <b>{safe_num(best_study["value"], 2)}{suffix}</b><br>
                        Attributes: {safe_int(best_study["attributes"])}<br>
                        Base: {safe_int(best_study["base"])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="card">
                    <div class="insight-title">Best Study</div>
                    <div class="insight-main">—</div>
                    <div class="insight-sub">Study data is not available for this view.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with c4:
        st.markdown(
            f"""
            <div class="card">
                <div class="insight-title">Tier Spread</div>
                <div class="insight-main">{spread_text}</div>
                <div class="insight-sub">
                    Difference between Top 25% and Bottom 25%.<br>
                    Wider spread means stronger discrimination.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# 10. CHARTS
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="section-title">Attribute Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top attributes by selected metric.</div>', unsafe_allow_html=True)

    chart_df = (
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

    if not chart_df.empty:
        fig = px.bar(
            chart_df.sort_values("value", ascending=True),
            x="value",
            y="parameter_name",
            orientation="h",
            color="parameter_group",
            hover_data=["base"],
            labels={
                "value": metric_label(selected_metric),
                "parameter_name": "",
                "parameter_group": "Group",
            },
            height=max(360, top_n * 32),
        )

        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=15, b=5),
            legend_title_text="Group",
        )

        st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Norm Tier</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average by benchmark tier.</div>', unsafe_allow_html=True)

    grade_df = (
        work
        .dropna(subset=["norm_grade", selected_metric])
        .groupby("norm_grade", observed=False)
        .agg(
            value=(selected_metric, "mean"),
            base=("base", "sum"),
            rows=("parameter_name", "count"),
        )
        .reset_index()
    )

    if not grade_df.empty:
        grade_df["norm_grade"] = grade_df["norm_grade"].astype(str)

        fig2 = px.bar(
            grade_df,
            x="norm_grade",
            y="value",
            hover_data=["base", "rows"],
            labels={
                "norm_grade": "",
                "value": metric_label(selected_metric),
            },
            height=360,
        )

        fig2.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=5, r=5, t=15, b=5),
            showlegend=False,
        )

        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 11. STUDY COVERAGE
# ============================================================

if "study" in df.columns and df["study"].notna().any():
    study_slice = df[df["slice_type"].astype(str).eq("study")].copy()

    if not study_slice.empty:
        st.markdown('<div class="section-title">Study Coverage</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Number of norm rows by study/project.</div>', unsafe_allow_html=True)

        study_df = (
            study_slice
            .dropna(subset=["study"])
            .groupby("study")
            .agg(
                rows=("parameter_name", "count"),
                attributes=("parameter_name", "nunique"),
                median_base=("base", "median"),
                avg_mean=("mean_score", "mean"),
            )
            .reset_index()
            .sort_values("study")
        )

        fig3 = px.bar(
            study_df,
            x="study",
            y="rows",
            hover_data=["attributes", "median_base", "avg_mean"],
            labels={"study": "", "rows": "Rows"},
            height=330,
        )

        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis_tickangle=-35,
            margin=dict(l=5, r=5, t=15, b=80),
            showlegend=False,
        )

        st.plotly_chart(fig3, use_container_width=True)


# ============================================================
# 12. NORM TABLE
# ============================================================

st.markdown('<div class="section-title">Norm Table</div>', unsafe_allow_html=True)

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
    height=360,
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
    "Top 25% indicates stronger benchmark performance; Bottom 25% indicates weaker benchmark performance. Interpret results together with base size."
)
