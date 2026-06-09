import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Deka Norm Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# DESIGN SYSTEM - DEKA INSIGHT STYLE
# ============================================================

DEKA_NAVY = "#0B1026"
DEKA_BLUE = "#1E2A4A"
DEKA_GOLD = "#F2A93B"
DEKA_CREAM = "#FAF8F2"
DEKA_GREY = "#EEF0F4"
DEKA_TEXT = "#222638"
DEKA_MUTED = "#6B7280"

st.markdown(
    f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background: {DEKA_CREAM};
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E5E7EB;
        }}

        .main .block-container {{
            padding-top: 2.2rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }}

        .hero-card {{
            background:
                radial-gradient(circle at 10% 10%, rgba(242,169,59,0.13), transparent 24%),
                linear-gradient(135deg, #FFFFFF 0%, #FBF7EF 100%);
            border: 1px solid #EFE6D6;
            border-radius: 28px;
            padding: 34px 38px;
            margin-bottom: 26px;
            box-shadow: 0 16px 40px rgba(11,16,38,0.08);
        }}

        .hero-eyebrow {{
            color: {DEKA_GOLD};
            font-weight: 800;
            letter-spacing: 0.08em;
            font-size: 0.82rem;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .hero-title {{
            color: {DEKA_NAVY};
            font-size: 2.5rem;
            line-height: 1.05;
            font-weight: 800;
            margin-bottom: 12px;
        }}

        .hero-subtitle {{
            color: {DEKA_BLUE};
            font-size: 1.05rem;
            line-height: 1.65;
            max-width: 880px;
        }}

        .section-title {{
            color: {DEKA_NAVY};
            font-size: 1.45rem;
            font-weight: 800;
            margin-top: 8px;
            margin-bottom: 12px;
        }}

        .section-caption {{
            color: {DEKA_MUTED};
            font-size: 0.94rem;
            margin-bottom: 14px;
        }}

        .metric-card {{
            background: #FFFFFF;
            border: 1px solid #E8E2D8;
            border-radius: 22px;
            padding: 22px 22px;
            min-height: 130px;
            box-shadow: 0 12px 28px rgba(11,16,38,0.06);
        }}

        .metric-label {{
            color: {DEKA_MUTED};
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 8px;
        }}

        .metric-value {{
            color: {DEKA_NAVY};
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.1;
        }}

        .metric-help {{
            color: {DEKA_MUTED};
            font-size: 0.82rem;
            margin-top: 8px;
        }}

        .insight-card {{
            background: #FFFFFF;
            border-left: 6px solid {DEKA_GOLD};
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 10px 22px rgba(11,16,38,0.05);
            margin-bottom: 12px;
        }}

        .insight-title {{
            color: {DEKA_NAVY};
            font-weight: 800;
            font-size: 1rem;
            margin-bottom: 6px;
        }}

        .insight-text {{
            color: {DEKA_BLUE};
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid #E5E7EB;
        }}

        .stButton > button {{
            background: {DEKA_NAVY};
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.65rem 1.2rem;
            font-weight: 700;
        }}

        .stDownloadButton > button {{
            background: {DEKA_GOLD};
            color: {DEKA_NAVY};
            border-radius: 999px;
            border: none;
            padding: 0.65rem 1.2rem;
            font-weight: 800;
        }}

        hr {{
            border: none;
            height: 1px;
            background: #E5E7EB;
            margin: 28px 0;
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
    query = "SELECT * FROM norm_value_all"
    df = pd.read_sql(query, engine)
    return df


df = load_data()

# ============================================================
# DATA PREPARATION
# ============================================================

# Convert numeric columns safely
numeric_cols = [
    "scale", "mean_score", "tb_pct", "t2b_pct", "t3b_pct",
    "base", "min_score", "max_score", "std_score"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Replace string None / nan style values
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace(["None", "nan", "NaN", ""], pd.NA)

# Grade order
grade_order = ["Top 25%", "Average 50%", "Bottom 25%"]
df["norm_grade"] = pd.Categorical(
    df["norm_grade"],
    categories=grade_order,
    ordered=True
)

# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("## Filters")
st.sidebar.caption("Gunakan filter untuk melihat norm berdasarkan segmentasi data.")

slice_options = sorted(df["slice_type"].dropna().unique().tolist())
default_slice = "Global" if "Global" in slice_options else slice_options[0]

selected_slice = st.sidebar.selectbox(
    "Slice Type",
    slice_options,
    index=slice_options.index(default_slice)
)

filtered = df[df["slice_type"] == selected_slice].copy()

# Dynamic slice value filters
slice_columns = selected_slice.split(" | ") if selected_slice != "Global" else []
slice_columns = [c for c in slice_columns if c in filtered.columns]

for col in slice_columns:
    options = sorted(filtered[col].dropna().astype(str).unique().tolist())

    if len(options) > 0:
        selected_values = st.sidebar.multiselect(
            col.replace("_", " ").title(),
            options,
            default=[]
        )

        if selected_values:
            filtered = filtered[filtered[col].astype(str).isin(selected_values)]

parameter_options = sorted(filtered["parameter_name"].dropna().unique().tolist())
selected_parameters = st.sidebar.multiselect(
    "Parameter",
    parameter_options,
    default=[]
)

if selected_parameters:
    filtered = filtered[filtered["parameter_name"].isin(selected_parameters)]

scale_options = sorted(filtered["scale"].dropna().unique().tolist())
selected_scales = st.sidebar.multiselect(
    "Scale",
    scale_options,
    default=scale_options
)

if selected_scales:
    filtered = filtered[filtered["scale"].isin(selected_scales)]

grade_options = [g for g in grade_order if g in filtered["norm_grade"].dropna().astype(str).unique()]
selected_grades = st.sidebar.multiselect(
    "Norm Grade",
    grade_options,
    default=grade_options
)

if selected_grades:
    filtered = filtered[filtered["norm_grade"].astype(str).isin(selected_grades)]

min_base = st.sidebar.number_input(
    "Minimum Base",
    min_value=0,
    value=10,
    step=10
)

filtered = filtered[filtered["base"] >= min_base].copy()

metric_view = st.sidebar.selectbox(
    "Main Metric",
    ["mean_score", "tb_pct", "t2b_pct", "t3b_pct"],
    index=0
)

top_n = st.sidebar.slider(
    "Top N Parameter",
    min_value=5,
    max_value=30,
    value=12,
    step=1
)

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-eyebrow">Deka Insight • Norm Database</div>
        <div class="hero-title">Survey Norm Intelligence Dashboard</div>
        <div class="hero-subtitle">
            Dashboard ini membantu membandingkan performa atribut survey terhadap historical norm.
            Output mencakup <b>Top 25%</b>, <b>Average 50%</b>, <b>Bottom 25%</b>, 
            serta metrik <b>Mean Score</b>, <b>Top Box</b>, <b>Top 2 Boxes</b>, dan <b>Top 3 Boxes</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# EMPTY STATE
# ============================================================

if filtered.empty:
    st.warning("Tidak ada data untuk kombinasi filter yang dipilih. Coba turunkan Minimum Base atau ubah filter.")
    st.stop()

# ============================================================
# KPI CALCULATION
# ============================================================

total_rows = len(filtered)
total_base = int(filtered["base"].sum())
avg_mean = filtered["mean_score"].mean()
avg_tb = filtered["tb_pct"].mean()
avg_t2b = filtered["t2b_pct"].mean()
avg_t3b = filtered["t3b_pct"].mean(skipna=True)

top_grade_df = filtered[filtered["norm_grade"].astype(str) == "Top 25%"].copy()
bottom_grade_df = filtered[filtered["norm_grade"].astype(str) == "Bottom 25%"].copy()

best_row = filtered.sort_values(metric_view, ascending=False).head(1)
weak_row = filtered.sort_values(metric_view, ascending=True).head(1)

# ============================================================
# KPI CARDS
# ============================================================

st.markdown('<div class="section-title">Executive Summary</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Ringkasan performa norm berdasarkan filter yang sedang dipilih.</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4, c5 = st.columns(5)

def metric_card(container, label, value, help_text):
    container.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

metric_card(c1, "Total Base", f"{total_base:,}", "Total respondent base dari hasil filter.")
metric_card(c2, "Mean Score", f"{avg_mean:.2f}", "Rata-rata skor across selected norm.")
metric_card(c3, "Avg TB%", f"{avg_tb:.1f}%", "Rata-rata Top Box.")
metric_card(c4, "Avg T2B%", f"{avg_t2b:.1f}%", "Rata-rata Top 2 Boxes.")
metric_card(c5, "Avg T3B%", "-" if pd.isna(avg_t3b) else f"{avg_t3b:.1f}%", "Top 3 Boxes untuk scale ≥ 7.")

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# INSIGHT CARDS
# ============================================================

st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Insight otomatis berdasarkan metrik utama yang dipilih.</div>',
    unsafe_allow_html=True
)

i1, i2, i3 = st.columns(3)

if not best_row.empty:
    r = best_row.iloc[0]
    i1.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Strongest Attribute</div>
            <div class="insight-text">
                <b>{r['parameter_name']}</b> memiliki nilai <b>{metric_view}</b> tertinggi 
                sebesar <b>{r[metric_view]:.2f}</b> pada norm <b>{r['norm_grade']}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if not weak_row.empty:
    r = weak_row.iloc[0]
    i2.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Improvement Area</div>
            <div class="insight-text">
                <b>{r['parameter_name']}</b> menjadi area yang perlu diperhatikan karena 
                memiliki <b>{metric_view}</b> terendah sebesar <b>{r[metric_view]:.2f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

i3.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Selected Segment</div>
        <div class="insight-text">
            Dashboard sedang menampilkan slice <b>{selected_slice}</b> dengan 
            <b>{total_rows:,}</b> baris norm value dan minimum base <b>{min_base}</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# CHART DATA
# ============================================================

chart_df = filtered.copy()

# For chart readability, aggregate by parameter + norm grade
chart_summary = (
    chart_df
    .groupby(["parameter_name", "scale", "norm_grade"], observed=False, dropna=False)
    .agg(
        mean_score=("mean_score", "mean"),
        tb_pct=("tb_pct", "mean"),
        t2b_pct=("t2b_pct", "mean"),
        t3b_pct=("t3b_pct", "mean"),
        base=("base", "sum")
    )
    .reset_index()
)

chart_summary = chart_summary.dropna(subset=[metric_view])
chart_summary = chart_summary.sort_values(metric_view, ascending=False).head(top_n * 3)

# ============================================================
# CHARTS
# ============================================================

left, right = st.columns([1.25, 1])

with left:
    st.markdown('<div class="section-title">Parameter Performance Ranking</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-caption">Ranking parameter berdasarkan <b>{metric_view}</b>.</div>',
        unsafe_allow_html=True
    )

    fig = px.bar(
        chart_summary,
        x=metric_view,
        y="parameter_name",
        color="norm_grade",
        orientation="h",
        hover_data=["scale", "base", "mean_score", "tb_pct", "t2b_pct", "t3b_pct"],
        color_discrete_map={
            "Top 25%": DEKA_GOLD,
            "Average 50%": DEKA_BLUE,
            "Bottom 25%": "#A7B0C0"
        }
    )

    fig.update_layout(
        height=560,
        plot_bgcolor=DEKA_CREAM,
        paper_bgcolor=DEKA_CREAM,
        font=dict(color=DEKA_TEXT),
        legend_title_text="Norm Grade",
        xaxis_title=metric_view.replace("_", " ").upper(),
        yaxis_title="",
        margin=dict(l=10, r=10, t=20, b=20)
    )

    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Norm Grade Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Perbandingan rata-rata metric antar norm grade.</div>',
        unsafe_allow_html=True
    )

    grade_summary = (
        filtered
        .groupby("norm_grade", observed=False)
        .agg(
            mean_score=("mean_score", "mean"),
            tb_pct=("tb_pct", "mean"),
            t2b_pct=("t2b_pct", "mean"),
            t3b_pct=("t3b_pct", "mean"),
            base=("base", "sum")
        )
        .reset_index()
    )

    fig2 = px.bar(
        grade_summary,
        x="norm_grade",
        y=metric_view,
        color="norm_grade",
        text=metric_view,
        color_discrete_map={
            "Top 25%": DEKA_GOLD,
            "Average 50%": DEKA_BLUE,
            "Bottom 25%": "#A7B0C0"
        }
    )

    fig2.update_traces(texttemplate="%{text:.2f}", textposition="outside")

    fig2.update_layout(
        height=560,
        plot_bgcolor=DEKA_CREAM,
        paper_bgcolor=DEKA_CREAM,
        font=dict(color=DEKA_TEXT),
        showlegend=False,
        xaxis_title="",
        yaxis_title=metric_view.replace("_", " ").upper(),
        margin=dict(l=10, r=10, t=20, b=20)
    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# GAP ANALYSIS
# ============================================================

st.markdown('<div class="section-title">Top vs Bottom Norm Gap Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Mengukur jarak performa antara Top 25% dan Bottom 25% untuk menemukan atribut yang paling membedakan.</div>',
    unsafe_allow_html=True
)

gap_source = filtered[filtered["norm_grade"].astype(str).isin(["Top 25%", "Bottom 25%"])].copy()

gap_pivot = gap_source.pivot_table(
    index=["parameter_name", "scale"],
    columns="norm_grade",
    values=metric_view,
    aggfunc="mean"
).reset_index()

if "Top 25%" in gap_pivot.columns and "Bottom 25%" in gap_pivot.columns:
    gap_pivot["gap"] = gap_pivot["Top 25%"] - gap_pivot["Bottom 25%"]
    gap_pivot = gap_pivot.sort_values("gap", ascending=False).head(top_n)

    fig3 = px.bar(
        gap_pivot,
        x="gap",
        y="parameter_name",
        orientation="h",
        hover_data=["scale", "Top 25%", "Bottom 25%"],
        color="gap",
        color_continuous_scale=[[0, "#E8E2D8"], [1, DEKA_GOLD]]
    )

    fig3.update_layout(
        height=480,
        plot_bgcolor=DEKA_CREAM,
        paper_bgcolor=DEKA_CREAM,
        font=dict(color=DEKA_TEXT),
        xaxis_title=f"Gap {metric_view}",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=20, b=20)
    )

    fig3.update_yaxes(autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Gap analysis membutuhkan Top 25% dan Bottom 25% dalam filter yang dipilih.")

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# DETAIL TABLE
# ============================================================

st.markdown('<div class="section-title">Detailed Norm Value Table</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Tabel final yang dapat digunakan untuk review, reporting, atau export.</div>',
    unsafe_allow_html=True
)

display_cols = [
    "slice_type",
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
    "sequence"
]

display_cols = [c for c in display_cols if c in filtered.columns]
table_df = filtered[display_cols].sort_values(
    ["parameter_name", "scale", "norm_grade"],
    ascending=[True, True, True]
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)

csv = table_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Norm Value CSV",
    data=csv,
    file_name="filtered_deka_norm_value.csv",
    mime="text/csv"
)

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Built with Python, PostgreSQL/Supabase, and Streamlit • Norm calculation: Top 25%, Average 50%, Bottom 25% by ranked respondent score."
)
