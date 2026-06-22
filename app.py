# ============================================================
# DEKA INSIGHT — PRODUCT TEST NORM DASHBOARD
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
PAGE_ICON = str(ICON_PATH) if ICON_PATH.exists() else "📊"

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. THEME — LIGHT / DARK MODE
# ============================================================

theme_base = st.get_option("theme.base") or "light"
IS_DARK = theme_base == "dark"

if IS_DARK:
    THEME = {
        "app_bg": "#0F1424",
        "app_bg_2": "#151B2E",
        "card_soft": "rgba(23,30,50,0.96)",
        "line": "#2B344F",
        "text": "#F4F6FB",
        "muted": "#A8B0C3",
        "hero_1": "#111A39",
        "hero_2": "#202B55",
        "sidebar": "#090F25",
        "grid": "#2D354D",
        "plot_bg": "rgba(0,0,0,0)",
        "input_bg": "#F7F8FF",
        "input_text": "#0B1026",
        "cream": "#1A2136",
        "gold": "#F2B85E",
        "navy": "#7C93C3",
        "blue": "#8AA8D8",
        "sage": "#8FBA9D",
        "sage_dark": "#74A384",
        "terracotta": "#D48A78",
        "plum": "#C49AC0",
        "teal": "#79C5C3",
        "olive": "#C3C979",
        "caramel": "#D9A25E",
        "stone": "#A7A0A0",
    }
else:
    THEME = {
        "app_bg": "#FAF7F0",
        "app_bg_2": "#FFFFFF",
        "card_soft": "rgba(255,255,255,0.96)",
        "line": "#E8E1D8",
        "text": "#0B1026",
        "muted": "#737B8E",
        "hero_1": "#111A39",
        "hero_2": "#192653",
        "sidebar": "#090F25",
        "grid": "#E9E1D6",
        "plot_bg": "rgba(0,0,0,0)",
        "input_bg": "#FFFFFF",
        "input_text": "#0B1026",
        "cream": "#F7EBD8",
        "gold": "#E3A93F",
        "navy": "#24304F",
        "blue": "#6F86A8",
        "sage": "#6F9278",
        "sage_dark": "#5F826C",
        "terracotta": "#C27763",
        "plum": "#8C6A86",
        "teal": "#5D9A9A",
        "olive": "#A0A86B",
        "caramel": "#B98246",
        "stone": "#8F8983",
    }

COLOR_TEXT = THEME["muted"]
COLOR_GRID = THEME["grid"]
COLOR_CREAM = THEME["cream"]

TIER_COLOR_MAP = {
    "Top 25%": THEME["sage_dark"],
    "Average 50%": THEME["gold"],
    "Bottom 25%": THEME["terracotta"],
}

BRAND_COLORS = {
    "Taste": THEME["gold"],
    "Aroma": THEME["blue"],
    "Appearance": THEME["plum"],
    "Liking": THEME["navy"],
    "Purchase Intent": THEME["terracotta"],
    "Aftertaste": THEME["teal"],
    "Amount / Topping": THEME["caramel"],
    "Texture": THEME["sage"],
    "Overall Taste": THEME["olive"],
    "Overall Attribute": THEME["olive"],
    "Other Attribute": THEME["stone"],
}

COLOR_SEQUENCE = [
    THEME["gold"],
    THEME["blue"],
    THEME["plum"],
    THEME["navy"],
    THEME["terracotta"],
    THEME["teal"],
    THEME["caramel"],
    THEME["sage"],
    THEME["olive"],
    THEME["stone"],
]

PLOT_CONFIG = {
    "displayModeBar": False,
    "responsive": True,
}


# ============================================================
# 3. HELPER FUNCTIONS
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
        "category": "Benchmark Category",
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
        "parameter_group": "Parameter Group",
        "scale": "Scale",
        "norm_grade": "Norm",
        "mean_score": "Mean",
        "tb_pct": "TB",
        "t2b_pct": "T2B",
        "t3b_pct": "T3B",
        "base": "Base",
    }
    return mapping.get(col, col.replace("_", " ").title())


def make_unique_columns(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col} ({seen[col] + 1})")
    return new_columns


def apply_plot_theme(fig, show_legend=True):
    fig.update_layout(
        plot_bgcolor=THEME["plot_bg"],
        paper_bgcolor=THEME["plot_bg"],
        margin=dict(l=5, r=5, t=15, b=5),
        font=dict(color=COLOR_TEXT),
        legend_title_text="Parameter Group",
        showlegend=show_legend,
        xaxis=dict(
            gridcolor=COLOR_GRID,
            zeroline=False,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            zeroline=False,
            tickfont=dict(color=COLOR_TEXT),
            title_font=dict(color=COLOR_TEXT),
        ),
    )
    return fig


def coverage_note_for_view(work, df, active_dims):
    if active_dims:
        primary_dim = active_dims[0]
        primary_label = col_label(primary_dim)
        primary_count = work[primary_dim].dropna().nunique() if primary_dim in work.columns else 0

        if primary_count > 0:
            return f"{safe_int(primary_count)} {primary_label.lower()}"

    study_slice = df[df["slice_type"].astype(str).eq("study")].copy()

    if "study" in study_slice.columns:
        study_count = study_slice["study"].dropna().nunique()
        if study_count > 0:
            return f"{safe_int(study_count)} studies"

    return f"{safe_int(len(work))} rows"


def html_step(number, title, body):
    return f"""
    <div class="method-card">
        <div class="step-number">{number}</div>
        <div class="method-title">{title}</div>
        <div class="method-body">{body}</div>
    </div>
    """


def html_info_card(title, body, badge=None):
    badge_html = f'<span class="mini-badge">{badge}</span>' if badge else ""
    return f"""
    <div class="info-card">
        <div class="info-title">{title}</div>
        <div class="info-body">{body}</div>
        {badge_html}
    </div>
    """


# ============================================================
# 4. CSS — DYNAMIC LIGHT / DARK
# ============================================================

st.markdown(
    f"""
    <style>
    :root {{
        --sidebar: {THEME["sidebar"]};
        --ink: {THEME["text"]};
        --muted: {THEME["muted"]};
        --gold: {THEME["gold"]};
        --card-soft: {THEME["card_soft"]};
        --line: {THEME["line"]};
        --bg: {THEME["app_bg"]};
        --bg2: {THEME["app_bg_2"]};
        --hero1: {THEME["hero_1"]};
        --hero2: {THEME["hero_2"]};
        --input-bg: {THEME["input_bg"]};
        --input-text: {THEME["input_text"]};
    }}

    .stApp {{
        background:
            radial-gradient(circle at top left, rgba(242,169,59,.08), transparent 28%),
            linear-gradient(180deg, var(--bg) 0%, var(--bg2) 44%, var(--bg) 100%);
        color: var(--ink);
    }}

    .block-container {{
        padding-top: 3.7rem;
        padding-bottom: 2.1rem;
        max-width: 1420px;
    }}

    section[data-testid="stSidebar"] {{
        background: var(--sidebar);
        border-right: 1px solid rgba(255,255,255,.08);
    }}

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {{
        color: #F7F8FF !important;
    }}

    section[data-testid="stSidebar"] .sidebar-logo {{
        text-align: center;
        margin: 8px 0 18px 0;
    }}

    section[data-testid="stSidebar"] .sidebar-logo img {{
        max-width: 136px;
    }}

    .filter-hint {{
        font-size: 12px;
        color: rgba(255,255,255,.62) !important;
        line-height: 1.35;
        margin: -2px 0 14px 0;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        border-radius: 12px !important;
        color: var(--input-text) !important;
        box-shadow: none !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] div,
    section[data-testid="stSidebar"] div[data-baseweb="select"] span {{
        color: var(--input-text) !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{
        fill: var(--input-text) !important;
        color: var(--input-text) !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
        background-color: var(--input-bg) !important;
        border: 1px solid rgba(255,255,255,.12) !important;
        border-radius: 12px !important;
        color: var(--input-text) !important;
    }}

    section[data-testid="stSidebar"] input {{
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
    }}

    div[data-baseweb="popover"] ul {{
        background-color: var(--input-bg) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(11,16,38,.10) !important;
    }}

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li span,
    div[data-baseweb="popover"] li div {{
        color: var(--input-text) !important;
        background-color: var(--input-bg) !important;
    }}

    div[data-baseweb="popover"] li:hover {{
        background-color: rgba(242,169,59,.18) !important;
        color: var(--input-text) !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] {{
        background-color: var(--gold) !important;
        border-radius: 999px !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] span {{
        color: #0B1026 !important;
        font-weight: 750 !important;
    }}

    section[data-testid="stSidebar"] div[data-baseweb="tag"] svg {{
        fill: #0B1026 !important;
    }}

    .hero {{
        background:
            linear-gradient(135deg, rgba(242,169,59,.08) 0%, rgba(242,169,59,0) 38%),
            linear-gradient(135deg, var(--hero1) 0%, var(--hero2) 100%);
        border-radius: 24px;
        padding: 24px 32px;
        color: white;
        box-shadow: 0 18px 44px rgba(9,15,37,.18);
        margin-top: 8px;
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,.08);
    }}

    .kicker {{
        color: var(--gold);
        font-size: 11px;
        font-weight: 900;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }}

    .hero-title {{
        font-size: 34px;
        font-weight: 900;
        line-height: 1.05;
        margin: 0 0 8px 0;
        color: #FFFFFF !important;
    }}

    .hero-sub {{
        color: rgba(255,255,255,.76);
        font-size: 14px;
        line-height: 1.48;
        max-width: 790px;
    }}

    .pill {{
        display: inline-block;
        margin-top: 13px;
        margin-right: 7px;
        padding: 6px 11px;
        border-radius: 999px;
        background: rgba(242,169,59,.13);
        border: 1px solid rgba(242,169,59,.34);
        color: var(--gold);
        font-size: 11px;
        font-weight: 850;
    }}

    .section-title {{
        font-size: 20px;
        font-weight: 900;
        color: var(--ink) !important;
        margin: 20px 0 6px 0;
    }}

    .section-sub {{
        color: var(--muted) !important;
        font-size: 13px;
        margin-bottom: 12px;
    }}

    .card {{
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 20px;
        padding: 17px 18px;
        box-shadow: 0 12px 30px rgba(11,16,38,.08);
        height: 100%;
    }}

    .kpi-card {{
        min-height: 132px;
    }}

    .insight-card {{
        min-height: 235px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}

    .metric-label {{
        color: var(--muted) !important;
        font-size: 10.5px;
        font-weight: 900;
        letter-spacing: .09em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }}

    .metric-value {{
        color: var(--ink) !important;
        font-size: 27px;
        font-weight: 900;
        line-height: 1.1;
    }}

    .metric-note {{
        color: var(--muted) !important;
        font-size: 12px;
        margin-top: 7px;
    }}

    .insight-title {{
        font-size: 12px;
        font-weight: 900;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: .06em;
        margin-bottom: 8px;
    }}

    .insight-main {{
        font-size: 20px;
        font-weight: 900;
        color: var(--ink) !important;
        line-height: 1.18;
        margin-bottom: 9px;
        min-height: 48px;
    }}

    .insight-sub {{
        color: var(--muted) !important;
        font-size: 12.5px;
        line-height: 1.48;
    }}

    .badge {{
        display: inline-block;
        padding: 5px 9px;
        border-radius: 999px;
        font-size: 11.5px;
        font-weight: 850;
        margin-top: 10px;
        width: fit-content;
    }}

    .badge-green {{
        background: rgba(111,146,120,.18);
        color: {THEME["sage_dark"]} !important;
    }}

    .badge-gold {{
        background: rgba(242,184,94,.18);
        color: {THEME["gold"]} !important;
    }}

    .badge-red {{
        background: rgba(194,119,99,.18);
        color: {THEME["terracotta"]} !important;
    }}

    .overview-hero {{
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 26px 30px;
        box-shadow: 0 14px 34px rgba(11,16,38,.08);
        margin-bottom: 18px;
    }}

    .overview-title {{
        font-size: 30px;
        font-weight: 950;
        color: var(--ink);
        margin-bottom: 8px;
    }}

    .overview-sub {{
        color: var(--muted);
        font-size: 14px;
        line-height: 1.55;
        max-width: 980px;
    }}

    .method-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin: 12px 0 18px 0;
    }}

    .method-card, .info-card {{
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 17px 18px;
        box-shadow: 0 10px 26px rgba(11,16,38,.055);
        min-height: 160px;
    }}

    .step-number {{
        display: inline-flex;
        width: 30px;
        height: 30px;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        background: rgba(242,184,94,.18);
        color: var(--gold);
        font-weight: 900;
        margin-bottom: 10px;
        font-size: 13px;
    }}

    .method-title, .info-title {{
        color: var(--ink);
        font-size: 15px;
        font-weight: 900;
        margin-bottom: 8px;
    }}

    .method-body, .info-body {{
        color: var(--muted);
        font-size: 12.5px;
        line-height: 1.55;
    }}

    .mini-badge {{
        display: inline-block;
        margin-top: 12px;
        padding: 5px 9px;
        border-radius: 999px;
        background: rgba(242,184,94,.16);
        color: var(--gold);
        font-size: 11px;
        font-weight: 850;
    }}

    .flow-box {{
        background: var(--card-soft);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 18px 20px;
        margin: 12px 0 18px 0;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }}

    .flow-text {{
        color: var(--ink);
        font-weight: 850;
    }}

    .mono-box {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        background: rgba(127,127,127,.08);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 16px 18px;
        color: var(--ink);
        font-size: 12.5px;
        line-height: 1.7;
        overflow-x: auto;
    }}

    div[data-testid="stDataFrame"] {{
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid var(--line);
    }}

    .stDownloadButton > button {{
        background: var(--sidebar) !important;
        color: white !important;
        border: 0 !important;
        border-radius: 999px !important;
        font-weight: 850 !important;
        padding: .55rem 1.2rem !important;
    }}

    .stButton > button {{
        border-radius: 999px !important;
        font-weight: 850 !important;
    }}

    @media (max-width: 980px) {{
        .method-grid {{
            grid-template-columns: 1fr;
        }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 5. DATABASE
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
# 6. CLEAN DATA
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

text_cols = [
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

for col in text_cols:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .replace(["", "nan", "NaN", "None", "none", "NULL", "null", "<NA>"], pd.NA)
    )

num_cols = [
    "actual_age",
    "num_of_product",
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

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

grade_order = ["Top 25%", "Average 50%", "Bottom 25%"]

df["norm_grade"] = pd.Categorical(
    df["norm_grade"],
    categories=grade_order,
    ordered=True,
)


# ============================================================
# 7. SIDEBAR FILTERS
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
        '<div class="filter-hint">Leave a filter empty to include all values.</div>',
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

    selected_groups = st.multiselect(
        "Parameter group",
        options=get_options(work, "parameter_group"),
        default=[],
        placeholder="All parameter groups",
    )

    work = filter_df(work, "parameter_group", selected_groups)

    selected_params = st.multiselect(
        "Parameter",
        options=get_options(work, "parameter_name"),
        default=[],
        placeholder="All parameters",
    )

    work = filter_df(work, "parameter_name", selected_params)

    st.caption("Parameter group narrows the parameter list. Parameter selects the item to benchmark.")

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

    norm_options = [
        x for x in grade_order
        if x in work["norm_grade"].astype(str).unique().tolist()
    ]

    selected_norms = st.multiselect(
        "Norm group",
        options=norm_options,
        default=[],
        placeholder="All norm groups",
    )

    if selected_norms:
        work = work[work["norm_grade"].astype(str).isin(selected_norms)].copy()

    min_base = st.number_input(
        "Minimum base",
        min_value=0,
        value=10,
        step=10,
    )

    work = work[work["base"].fillna(0) >= min_base].copy()

    top_n = st.slider(
        "Top parameters",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
    )

    st.markdown("---")

    if st.button("Refresh data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()


# ============================================================
# 8. HERO
# ============================================================

st.markdown(
    f"""
    <div class="hero">
        <div class="kicker">DEKA INSIGHT NORM DATABASE</div>
        <div class="hero-title">Product Test Norm Dashboard.</div>
        <div class="hero-sub">
            Compare product-test results with historical benchmarks across studies, segments, methods, and parameters.
        </div>
        <span class="pill">{selected_slice_label}</span>
        <span class="pill">{selected_metric_label}</span>
        <span class="pill">{safe_int(len(work))} norm rows</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if work.empty:
    st.warning("No data for the selected filters. Remove a filter or lower the minimum base.")
    st.stop()


# ============================================================
# 9. DERIVED VALUES
# ============================================================

metric_df = work.dropna(subset=[selected_metric]).copy()
suffix = metric_suffix(selected_metric)

coverage_note = coverage_note_for_view(work, df, active_dims)
parameter_count = work["parameter_name"].dropna().nunique()
parameter_group_count = work["parameter_group"].dropna().nunique()
slice_count = df["slice_type"].dropna().nunique()
study_count_all = df.loc[df["slice_type"].astype(str).eq("study"), "study"].dropna().nunique()
category_count_all = df.loc[df["slice_type"].astype(str).eq("category"), "category"].dropna().nunique()
median_base = work["base"].median()
avg_metric = metric_df[selected_metric].mean()

tier_summary_for_kpi = (
    metric_df
    .dropna(subset=["norm_grade"])
    .groupby("norm_grade", observed=False)
    .agg(value=(selected_metric, "mean"))
    .reset_index()
)

top_tier_mean = np.nan
bottom_tier_mean = np.nan
tier_gap = np.nan

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


# ============================================================
# 10. TABS
# ============================================================

tab_overview, tab_dashboard, tab_table = st.tabs(
    ["Overview", "Dashboard", "Norm Table"]
)


# ============================================================
# 10A. OVERVIEW TAB
# ============================================================

with tab_overview:
    st.markdown(
        """
        <div class="overview-hero">
            <div class="kicker">SYSTEM OVERVIEW</div>
            <div class="overview-title">Deka Insight Norm Database Methodology</div>
            <div class="overview-sub">
                This page explains how the product-test norm database is constructed, starting from raw Excel sheets,
                continuing through preprocessing, metadata extraction, parameter standardization, norm-tier formation,
                descriptive statistical analysis, and dashboard-ready database modeling.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    o1, o2, o3, o4, o5 = st.columns(5)

    overview_kpis = [
        ("Norm Rows", safe_int(len(df)), "Aggregated benchmark records"),
        ("Benchmark Views", safe_int(slice_count), "Available analytical slices"),
        ("Studies", safe_int(study_count_all), "Study / project sources"),
        ("Parameters", safe_int(df["parameter_name"].dropna().nunique()), "Standardized items"),
        ("Parameter Groups", safe_int(df["parameter_group"].dropna().nunique()), "Analytical groupings"),
    ]

    for col, (label, value, note) in zip([o1, o2, o3, o4, o5], overview_kpis):
        with col:
            st.markdown(
                f"""
                <div class="card kpi-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">1. Data Processing Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The system transforms raw product-test survey data into a normalized and benchmark-ready analytical database.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="method-grid">
            {html_step("01", "Raw Data Import", "Each Excel sheet represents a study or project. Metadata columns are retained, while product-test parameter columns are prepared for transformation.")}
            {html_step("02", "Metadata Separation", "Respondent and study-level fields such as study, category, gender, age, SES, methodology, test type, and sequence are separated from scoring variables.")}
            {html_step("03", "Wide-to-Long Transformation", "Parameter columns are melted into long format so that each row represents one respondent score for one parameter.")}
            {html_step("04", "Parameter Cleaning", "Parameter names are standardized by removing inconsistent spacing, repeated symbols, scale suffixes, and alternative writing styles.")}
            {html_step("05", "Parameter Reduction", "Duplicated or semantically similar parameter labels are consolidated into a single canonical parameter name and parameter key.")}
            {html_step("06", "Parameter Grouping", "Each parameter is assigned to a parameter group such as Taste, Aroma, Appearance, Texture, Liking, Purchase Intent, Aftertaste, or Other Attribute.")}
            {html_step("07", "Descriptive Statistics", "Mean score, Top Box, Top 2 Boxes, Top 3 Boxes, base size, minimum score, maximum score, and standard deviation are computed.")}
            {html_step("08", "Norm Tier Formation", "Scores are ranked into Top 25%, Average 50%, and Bottom 25% to create a benchmark tier for each parameter and slice.")}
            {html_step("09", "Dashboard Aggregation", "Final summary tables are stored in Supabase and queried by Streamlit for fast filtering and visualization.")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">2. End-to-End Data Flow</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="flow-box">
            <span class="flow-text">Excel Sheets</span>
            &nbsp;→&nbsp; metadata extraction
            &nbsp;→&nbsp; wide-to-long transformation
            &nbsp;→&nbsp; parameter standardization
            &nbsp;→&nbsp; parameter grouping
            &nbsp;→&nbsp; descriptive statistics
            &nbsp;→&nbsp; norm tiering
            &nbsp;→&nbsp; database tables
            &nbsp;→&nbsp; Streamlit dashboard
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        st.markdown('<div class="section-title">3. Metadata Structure</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Metadata is used as segmentation variables for slicing benchmark results.</div>',
            unsafe_allow_html=True,
        )

        metadata_df = pd.DataFrame(
            [
                ["study", "Study / project identifier", "Source sheet or project name"],
                ["category", "Benchmark category", "Main product category"],
                ["sub_category", "Sub-category", "More specific product category"],
                ["detail_product", "Product detail", "Detailed product description"],
                ["gender", "Respondent gender", "Demographic segmentation"],
                ["actual_age", "Actual age", "Raw respondent age"],
                ["age_group", "Age group", "Grouped age segmentation"],
                ["ses", "Socio-economic status", "Consumer profile segmentation"],
                ["occupation", "Occupation", "Respondent occupation"],
                ["type_of_study", "Type of study", "Study classification"],
                ["test_type", "Test type", "Product-test setting"],
                ["methodology", "Methodology", "Testing method"],
                ["sub_method", "Sub-method", "Detailed method classification"],
                ["num_of_product", "# of product", "Number of tested products"],
                ["sequence", "Sequence", "Product test order"],
            ],
            columns=["Field", "Meaning", "Usage"],
        )

        st.dataframe(metadata_df, use_container_width=True, hide_index=True, height=420)

    with c2:
        st.markdown('<div class="section-title">4. Parameter Processing</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Parameter standardization prevents fragmented analysis caused by inconsistent label writing.</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="flow-box">
                <b>Parameter reduction</b> in this dashboard does not mean dimensionality reduction such as PCA.
                It refers to cleaning and consolidating duplicated or inconsistent parameter labels into a consistent
                parameter master. For example, labels with spacing differences, punctuation differences, or repeated
                scale suffixes are mapped into one standardized parameter name.
                <br><br>
                <b>Parameter grouping</b> then assigns each cleaned parameter into an analytical group.
                This makes the dashboard easier to interpret because users can compare broad groups such as Taste,
                Aroma, Appearance, Texture, Liking, Purchase Intent, Aftertaste, and Other Attribute.
            </div>
            """,
            unsafe_allow_html=True,
        )

        parameter_summary = (
            df.groupby("parameter_group", dropna=False)
            .agg(
                parameters=("parameter_name", "nunique"),
                norm_rows=("parameter_name", "count"),
                avg_base=("base", "mean"),
            )
            .reset_index()
            .sort_values("parameters", ascending=False)
            .rename(
                columns={
                    "parameter_group": "Parameter Group",
                    "parameters": "Unique Parameters",
                    "norm_rows": "Norm Rows",
                    "avg_base": "Average Base",
                }
            )
        )

        st.dataframe(parameter_summary, use_container_width=True, hide_index=True, height=250)

    st.markdown('<div class="section-title">5. Descriptive Statistical Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The dashboard uses descriptive and benchmark-based statistics to summarize product-test performance.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="method-grid">
            {html_info_card("Mean Score", "Average score of a parameter within the selected benchmark slice. This is the primary central tendency indicator for product-test performance.", "Descriptive")}
            {html_info_card("Base Size", "Number of valid responses included in a parameter-level calculation. Larger base size indicates more stable interpretation.", "Reliability")}
            {html_info_card("Top Box", "Percentage of respondents giving the maximum score on the scale. For a 7-point scale, Top Box represents score 7.", "Preference strength")}
            {html_info_card("Top 2 Boxes", "Percentage of respondents giving the two highest scores. For a 7-point scale, this includes scores 6 and 7.", "Positive response")}
            {html_info_card("Top 3 Boxes", "Percentage of respondents giving the three highest scores. For a 7-point scale, this includes scores 5, 6, and 7.", "Broader positivity")}
            {html_info_card("Tier Gap", "Difference between Top 25% and Bottom 25% benchmark tier averages. Higher gap indicates stronger parameter discrimination.", "Benchmark spread")}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">6. Norm Tier Construction</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="flow-box">
            Norm tiering is used to interpret scores relatively within the norm database. For each parameter and analytical slice,
            score distributions are divided into three benchmark tiers:
            <br><br>
            <b>Top 25%</b> represents the stronger benchmark tier, <b>Average 50%</b> represents the middle benchmark range,
            and <b>Bottom 25%</b> represents the weaker benchmark tier. These labels do not always mean the absolute score is
            good or bad; instead, they describe the score position relative to other records in the norm database.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">7. Database Normalization and Analytical Tables</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">The database design separates raw responses, parameter master data, and aggregated dashboard-ready norm values.</div>',
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)

    with d1:
        st.markdown(
            html_info_card(
                "fact_response",
                "Stores respondent-level long-format score data. Each row represents one respondent score for one parameter with its metadata.",
                "Fact table"
            ),
            unsafe_allow_html=True,
        )

    with d2:
        st.markdown(
            html_info_card(
                "dim_parameter",
                "Stores standardized parameter information such as parameter ID, parameter name, parameter key, parameter group, and scale.",
                "Dimension table"
            ),
            unsafe_allow_html=True,
        )

    with d3:
        st.markdown(
            html_info_card(
                "norm_value_all",
                "Stores aggregated benchmark values across slice types. This table supports fast dashboard queries and dynamic filtering.",
                "Analytical table"
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="mono-box">
        dim_parameter 1 ─── * fact_response<br>
        dim_parameter 1 ─── * norm_base<br>
        dim_parameter 1 ─── * norm_value_all<br><br>
        fact_response     = respondent-level score data<br>
        dim_parameter     = standardized parameter master<br>
        norm_base         = parameter-level benchmark tier base<br>
        norm_value_all    = dashboard-ready aggregated norm table
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">8. Dashboard Output</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="method-grid">
            {html_info_card("Benchmark Snapshot", "Provides high-level coverage, number of parameters, base quality, market average, and tier gap for the selected view.", "KPI")}
            {html_info_card("Key Insights", "Highlights leading parameter, improvement area, leading segment, and lowest segment based on the selected metric.", "Insight")}
            {html_info_card("Parameter Ranking", "Ranks parameters by selected metric to identify the strongest product-test attributes.", "Ranking")}
            {html_info_card("Norm Tier Profile", "Compares average performance across Top 25%, Average 50%, and Bottom 25% benchmark tiers.", "Tier")}
            {html_info_card("Parameter Discrimination", "Shows the difference between upper and lower benchmark tiers to identify parameters that best separate performance levels.", "Gap")}
            {html_info_card("Norm Table", "Provides detailed filtered records for review and CSV export.", "Export")}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10B. DASHBOARD TAB
# ============================================================

with tab_dashboard:
    st.markdown('<div class="section-title">Benchmark Snapshot</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    kpis = [
        ("Norm Coverage", safe_int(len(work)), coverage_note),
        ("Parameters", safe_int(parameter_count), "Compared items"),
        ("Base Quality", safe_int(median_base), confidence(median_base)),
        ("Market Avg", f"{safe_num(avg_metric, 2)}{suffix}", selected_metric_label),
        ("Tier Gap", f"{safe_num(tier_gap, 2)}{suffix}", "Top vs bottom tier"),
    ]

    for col, (label, value, note) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(
                f"""
                <div class="card kpi-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)

    if not metric_df.empty:
        insight_base_threshold = max(min_base, 30)

        insight_df = metric_df[
            metric_df["base"].fillna(0) >= insight_base_threshold
        ].copy()

        if insight_df.empty:
            insight_df = metric_df.copy()

        param_summary = (
            insight_df
            .groupby(["parameter_name", "parameter_group"], dropna=False)
            .agg(
                value=(selected_metric, "mean"),
                base=("base", "sum"),
                mean_score=("mean_score", "mean"),
            )
            .reset_index()
        )

        best_param = param_summary.sort_values("value", ascending=False).iloc[0]
        weak_param = param_summary.sort_values("value", ascending=True).iloc[0]

        segment_label = None
        best_segment = None
        weak_segment = None

        if active_dims:
            segment_col = active_dims[0]

            if segment_col in insight_df.columns and insight_df[segment_col].notna().any():
                segment_label = col_label(segment_col)

                segment_summary = (
                    insight_df
                    .dropna(subset=[segment_col])
                    .groupby(segment_col)
                    .agg(
                        value=(selected_metric, "mean"),
                        base=("base", "sum"),
                        parameters=("parameter_name", "nunique"),
                    )
                    .reset_index()
                    .rename(columns={segment_col: "segment"})
                )

                if not segment_summary.empty:
                    best_segment = segment_summary.sort_values("value", ascending=False).iloc[0]
                    weak_segment = segment_summary.sort_values("value", ascending=True).iloc[0]

        if best_segment is None:
            study_df_for_insight = df[df["slice_type"].astype(str).eq("study")].copy()

            if not study_df_for_insight.empty:
                study_df_for_insight = study_df_for_insight.dropna(subset=["study", selected_metric])

                if not study_df_for_insight.empty:
                    segment_label = "Study / Project"

                    segment_summary = (
                        study_df_for_insight
                        .groupby("study")
                        .agg(
                            value=(selected_metric, "mean"),
                            base=("base", "sum"),
                            parameters=("parameter_name", "nunique"),
                        )
                        .reset_index()
                        .rename(columns={"study": "segment"})
                    )

                    best_segment = segment_summary.sort_values("value", ascending=False).iloc[0]
                    weak_segment = segment_summary.sort_values("value", ascending=True).iloc[0]

        i1, i2, i3, i4 = st.columns(4)

        with i1:
            st.markdown(
                f"""
                <div class="card insight-card">
                    <div>
                        <div class="insight-title">Leading Parameter</div>
                        <div class="insight-main">{clean_value(best_param["parameter_name"])}</div>
                        <div class="insight-sub">
                            {selected_metric_label}: <b>{safe_num(best_param["value"], 2)}{suffix}</b><br>
                            Parameter Group: {clean_value(best_param["parameter_group"])}<br>
                            Base: {safe_int(best_param["base"])}
                        </div>
                    </div>
                    <span class="badge badge-green">Strength</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with i2:
            st.markdown(
                f"""
                <div class="card insight-card">
                    <div>
                        <div class="insight-title">Improvement Area</div>
                        <div class="insight-main">{clean_value(weak_param["parameter_name"])}</div>
                        <div class="insight-sub">
                            {selected_metric_label}: <b>{safe_num(weak_param["value"], 2)}{suffix}</b><br>
                            Parameter Group: {clean_value(weak_param["parameter_group"])}<br>
                            Base: {safe_int(weak_param["base"])}
                        </div>
                    </div>
                    <span class="badge badge-red">Watch</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with i3:
            if best_segment is not None:
                st.markdown(
                    f"""
                    <div class="card insight-card">
                        <div>
                            <div class="insight-title">Leading {segment_label}</div>
                            <div class="insight-main">{clean_value(best_segment["segment"])}</div>
                            <div class="insight-sub">
                                {selected_metric_label}: <b>{safe_num(best_segment["value"], 2)}{suffix}</b><br>
                                Parameters: {safe_int(best_segment["parameters"])}<br>
                                Base: {safe_int(best_segment["base"])}
                            </div>
                        </div>
                        <span class="badge badge-green">Lead</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with i4:
            if weak_segment is not None:
                st.markdown(
                    f"""
                    <div class="card insight-card">
                        <div>
                            <div class="insight-title">Lowest {segment_label}</div>
                            <div class="insight-main">{clean_value(weak_segment["segment"])}</div>
                            <div class="insight-sub">
                                {selected_metric_label}: <b>{safe_num(weak_segment["value"], 2)}{suffix}</b><br>
                                Parameters: {safe_int(weak_segment["parameters"])}<br>
                                Base: {safe_int(weak_segment["base"])}
                            </div>
                        </div>
                        <span class="badge badge-gold">Compare</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown('<div class="section-title">Parameter Ranking</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Highest-performing parameters for the selected benchmark.</div>',
            unsafe_allow_html=True,
        )

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
                color_discrete_map=BRAND_COLORS,
                color_discrete_sequence=COLOR_SEQUENCE,
                hover_data=["base"],
                labels={
                    "value": selected_metric_label,
                    "parameter_name": "",
                    "parameter_group": "Parameter Group",
                },
                height=max(350, top_n * 32),
            )

            fig_rank.update_traces(
                marker_line_color=COLOR_CREAM,
                marker_line_width=0.8,
                opacity=0.94,
            )

            apply_plot_theme(fig_rank, show_legend=True)

            st.plotly_chart(
                fig_rank,
                use_container_width=True,
                config=PLOT_CONFIG,
            )

    with right:
        st.markdown('<div class="section-title">Norm Tier Profile</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Average performance by benchmark tier.</div>',
            unsafe_allow_html=True,
        )

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
                color="norm_grade",
                color_discrete_map=TIER_COLOR_MAP,
                hover_data=["base", "rows"],
                labels={
                    "norm_grade": "",
                    "value": selected_metric_label,
                },
                height=350,
            )

            fig_tier.update_traces(
                marker_line_color=COLOR_CREAM,
                marker_line_width=0.8,
                opacity=0.94,
            )

            fig_tier.update_layout(
                plot_bgcolor=THEME["plot_bg"],
                paper_bgcolor=THEME["plot_bg"],
                margin=dict(l=5, r=5, t=15, b=5),
                showlegend=False,
                font=dict(color=COLOR_TEXT),
                yaxis=dict(
                    gridcolor=COLOR_GRID,
                    zeroline=False,
                    tickfont=dict(color=COLOR_TEXT),
                    title_font=dict(color=COLOR_TEXT),
                ),
                xaxis=dict(
                    gridcolor="rgba(0,0,0,0)",
                    zeroline=False,
                    tickfont=dict(color=COLOR_TEXT),
                    title_font=dict(color=COLOR_TEXT),
                ),
            )

            st.plotly_chart(
                fig_tier,
                use_container_width=True,
                config=PLOT_CONFIG,
            )

    left2, right2 = st.columns([1.2, 1])

    with left2:
        st.markdown('<div class="section-title">Parameter Discrimination</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-sub">Gap between top and bottom benchmark tiers by parameter.</div>',
            unsafe_allow_html=True,
        )

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

            gap_df = (
                gap_df
                .dropna(subset=["gap"])
                .sort_values("gap", ascending=False)
                .head(top_n)
            )

            if not gap_df.empty:
                fig_gap = px.bar(
                    gap_df.sort_values("gap", ascending=True),
                    x="gap",
                    y="parameter_name",
                    orientation="h",
                    color="parameter_group",
                    color_discrete_map=BRAND_COLORS,
                    color_discrete_sequence=COLOR_SEQUENCE,
                    hover_data=["Top 25%", "Bottom 25%"],
                    labels={
                        "gap": f"Tier Gap ({selected_metric_label})",
                        "parameter_name": "",
                        "parameter_group": "Parameter Group",
                    },
                    height=max(350, top_n * 32),
                )

                fig_gap.update_traces(
                    marker_line_color=COLOR_CREAM,
                    marker_line_width=0.8,
                    opacity=0.94,
                )

                apply_plot_theme(fig_gap, show_legend=True)

                st.plotly_chart(
                    fig_gap,
                    use_container_width=True,
                    config=PLOT_CONFIG,
                )

    with right2:
        if active_dims:
            segment_col = active_dims[0]
            segment_title = f"{col_label(segment_col)} Performance"
            segment_subtitle = f"Average benchmark performance by {col_label(segment_col).lower()}."

            segment_perf = (
                metric_df
                .dropna(subset=[segment_col])
                .groupby(segment_col)
                .agg(
                    value=(selected_metric, "mean"),
                    base=("base", "sum"),
                    parameters=("parameter_name", "nunique"),
                )
                .reset_index()
                .rename(columns={segment_col: "segment"})
                .sort_values("value", ascending=False)
            )

        else:
            segment_title = "Study Performance"
            segment_subtitle = "Average benchmark performance by study/project."

            study_slice = df[df["slice_type"].astype(str).eq("study")].copy()

            segment_perf = (
                study_slice
                .dropna(subset=["study", selected_metric])
                .groupby("study")
                .agg(
                    value=(selected_metric, "mean"),
                    base=("base", "sum"),
                    parameters=("parameter_name", "nunique"),
                )
                .reset_index()
                .rename(columns={"study": "segment"})
                .sort_values("value", ascending=False)
            )

        st.markdown(f'<div class="section-title">{segment_title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-sub">{segment_subtitle}</div>', unsafe_allow_html=True)

        if not segment_perf.empty:
            if len(segment_perf) <= 8:
                fig_segment = px.bar(
                    segment_perf.sort_values("value", ascending=False),
                    x="segment",
                    y="value",
                    hover_data=["base", "parameters"],
                    labels={
                        "segment": "",
                        "value": selected_metric_label,
                    },
                    height=350,
                )

                fig_segment.update_traces(
                    marker_color=THEME["blue"],
                    marker_line_color=THEME["gold"],
                    marker_line_width=0.8,
                    opacity=0.9,
                )

                fig_segment.update_layout(
                    plot_bgcolor=THEME["plot_bg"],
                    paper_bgcolor=THEME["plot_bg"],
                    margin=dict(l=5, r=5, t=15, b=55),
                    font=dict(color=COLOR_TEXT),
                    yaxis=dict(
                        gridcolor=COLOR_GRID,
                        zeroline=False,
                        tickfont=dict(color=COLOR_TEXT),
                        title_font=dict(color=COLOR_TEXT),
                    ),
                    xaxis=dict(
                        gridcolor="rgba(0,0,0,0)",
                        tickangle=-25,
                        zeroline=False,
                        tickfont=dict(color=COLOR_TEXT),
                        title_font=dict(color=COLOR_TEXT),
                    ),
                )

            else:
                fig_segment = px.scatter(
                    segment_perf,
                    x="base",
                    y="value",
                    size="parameters",
                    hover_name="segment",
                    labels={
                        "base": "Total Base",
                        "value": selected_metric_label,
                        "parameters": "Parameters",
                    },
                    height=350,
                )

                fig_segment.update_traces(
                    marker=dict(
                        color=THEME["blue"],
                        line=dict(width=1.3, color=THEME["gold"]),
                        opacity=0.82,
                    )
                )

                fig_segment.update_layout(
                    plot_bgcolor=THEME["plot_bg"],
                    paper_bgcolor=THEME["plot_bg"],
                    margin=dict(l=5, r=5, t=15, b=5),
                    font=dict(color=COLOR_TEXT),
                    xaxis=dict(
                        gridcolor=COLOR_GRID,
                        zeroline=False,
                        tickfont=dict(color=COLOR_TEXT),
                        title_font=dict(color=COLOR_TEXT),
                    ),
                    yaxis=dict(
                        gridcolor=COLOR_GRID,
                        zeroline=False,
                        tickfont=dict(color=COLOR_TEXT),
                        title_font=dict(color=COLOR_TEXT),
                    ),
                )

            st.plotly_chart(
                fig_segment,
                use_container_width=True,
                config=PLOT_CONFIG,
            )


# ============================================================
# 10C. NORM TABLE TAB
# ============================================================

with tab_table:
    st.markdown('<div class="section-title">Norm Table</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-sub">Filtered benchmark table for export or detailed review.</div>',
        unsafe_allow_html=True,
    )

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
        if not pd.api.types.is_numeric_dtype(table[col]):
            table[col] = (
                table[col]
                .astype("string")
                .fillna("—")
                .replace(["nan", "None", "<NA>", "null", "NaT"], "—")
            )

    table = table.rename(columns=col_label)
    table.columns = make_unique_columns(table.columns)

    st.dataframe(
        table.head(300),
        use_container_width=True,
        hide_index=True,
        height=430,
    )

    st.caption(f"Showing 300 of {len(table):,} filtered rows.")

    st.download_button(
        "Download CSV",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="deka_norm_filtered.csv",
        mime="text/csv",
    )
