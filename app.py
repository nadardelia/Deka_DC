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
    page_icon="🟡",
    layout="wide"
)

# ============================================================
# COLOR PALETTE
# Inspired by Deka Insight website
# ============================================================

NAVY = "#0B1026"
BLUE = "#1E2A4A"
GOLD = "#F2A93B"
CREAM = "#FAF8F2"
CARD = "#FFFFFF"
SOFT = "#F4EFE7"
GREY = "#747B8D"
LINE = "#E7E0D6"
MUTED_GOLD = "#F7D9A2"

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
                radial-gradient(circle at 12% 8%, rgba(242,169,59,0.10), transparent 22%),
                linear-gradient(180deg, #FFFCF7 0%, {CREAM} 100%);
            color: {NAVY};
        }}

        .main .block-container {{
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }}

        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #ECE7DD;
        }}

        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {NAVY};
            font-weight: 900;
        }}

        .hero {{
            position: relative;
            background:
                radial-gradient(circle at top left, rgba(242,169,59,0.18), transparent 28%),
                linear-gradient(135deg, #FFFFFF 0%, #FFF9EE 100%);
            border: 1px solid #EFE1CB;
            border-radius: 34px;
            padding: 46px 52px;
            box-shadow: 0 28px 70px rgba(11,16,38,0.08);
            margin-bottom: 36px;
            overflow: hidden;
        }}

        .hero:after {{
            content: "";
            position: absolute;
            right: -80px;
            top: -80px;
            width: 260px;
            height: 260px;
            background: rgba(242,169,59,0.10);
            border-radius: 50%;
        }}

        .eyebrow {{
            font-size: 0.78rem;
            letter-spacing: 0.14em;
            color: {GOLD};
            font-weight: 900;
            text-transform: uppercase;
            margin-bottom: 16px;
        }}

        .hero-title {{
            font-size: 3.2rem;
            line-height: 1.02;
            color: {NAVY};
            font-weight: 900;
            letter-spacing: -0.055em;
            margin-bottom: 18px;
            max-width: 980px;
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
            font-size: 0.74rem;
            color: {GREY};
            font-weight: 900;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }}

        .metric-value {{
            color: {NAVY};
            font-size: 2.05rem;
            font-weight: 900;
            letter-spacing: -0.045em;
            line-height: 1;
        }}

        .metric-note {{
            color: {GREY};
            font-size: 0.80rem;
            line-height: 1.35;
        }}

        .insight-card {{
            background: {CARD};
            border: 1px solid {LINE};
            border-left: 7px solid {GOLD};
            border-radius: 24px;
            padding: 22px 24px;
            box-shadow: 0 16px 34px rgba(11,16,38,0.06);
            min-height: 142px;
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
            font-size: 0.98rem;
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
# DATA CLEANING FOR DASHBOARD
# ============================================================

numeric_cols = [
    "scale", "mean_score", "tb_pct", "t2b_pct", "t3b_pct",
    "base", "min_score", "max_score", "std_score"
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = df[col].replace(["None", "nan", "NaN", "", "NULL"], pd.NA)

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
st.sidebar.caption("Choose a segment, then narrow the benchmark.")

slice_options = sorted(df["slice_type"].dropna().unique().tolist())
default_slice = "Global" if "Global" in slice_options else slice_options[0]

selected_slice = st.sidebar.selectbox(
    "Benchmark view",
    slice_options,
    index=slice_options.index(default_slice)
)

filtered = df[df["slice_type"] == selected_slice].copy()

slice_columns = selected_slice.split(" | ") if selected_slice != "Global" else []
slice_columns = [c for c in slice_columns if c in filtered.columns]

for col in slice_columns:
    values = sorted(filtered[col].dropna().astype(str).unique().tolist())

    if values:
        chosen = st.sidebar.multiselect(
            col.replace("_", " ").title(),
            values,
            default=[]
        )

        if chosen:
            filtered = filtered[filtered[col].astype(str).isin(chosen)]

parameters = sorted(filtered["parameter_name"].dropna().unique().tolist())
chosen_params = st.sidebar.multiselect(
    "Parameter",
    parameters,
    default=[]
)

if chosen_params:
    filtered = filtered[filtered["parameter_name"].isin(chosen_params)]

scales = sorted(filtered["scale"].dropna().unique().tolist())
chosen_scales = st.sidebar.multiselect(
    "Scale",
    scales,
    default=scales
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
    default=grades
)

if chosen_grades:
    filtered = filtered[filtered["norm_grade"].astype(str).isin(chosen_grades)]

min_base = st.sidebar.number_input(
    "Minimum base",
    min_value=0,
    value=10,
    step=10
)

filtered = filtered[filtered["base"] >= min_base].copy()

metric_options = {
    "Mean Score": "mean_score",
    "Top Box": "tb_pct",
    "Top 2 Boxes": "t2b_pct",
    "Top 3 Boxes": "t3b_pct"
}

metric_label = st.sidebar.selectbox(
    "Focus metric",
    list(metric_options.keys())
)

metric_col = metric_options[metric_label]

top_n = st.sidebar.slider(
    "Show top parameters",
    min_value=5,
    max_value=25,
    value=12
)

if filtered.empty:
    st.warning("No benchmark found for this selection. Try lowering the base or changing the segment.")
    st.stop()

# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">DEKA INSIGHT • SURVEY NORM DATABASE</div>
        <div class="hero-title">
            Delivering Insights<br>
            <span class="hero-accent">Since 1995</span>
        </div>
        <div class="hero-copy">
            Indonesia's leading independent market research agency with deep cultural understanding and international expertise.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# KPI SNAPSHOT
# ============================================================

total_base = int(filtered["base"].sum())
avg_mean = filtered["mean_score"].mean()
avg_tb = filtered["tb_pct"].mean()
avg_t2b = filtered["t2b_pct"].mean()
avg_t3b = filtered["t3b_pct"].mean(skipna=True)

st.markdown('<div class="section-title">Benchmark snapshot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">A clean read of the selected norm universe.</div>',
    unsafe_allow_html=True
)

k1, k2, k3, k4, k5 = st.columns(5)

def render_metric(col, label, value, note):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

render_metric(k1, "Base", f"{total_base:,}", "Total responses")
render_metric(k2, "Mean", f"{avg_mean:.2f}", "Average score")
render_metric(k3, "TB", f"{avg_tb:.1f}%", "Top Box")
render_metric(k4, "T2B", f"{avg_t2b:.1f}%", "Top 2 Boxes")
render_metric(k5, "T3B", "-" if pd.isna(avg_t3b) else f"{avg_t3b:.1f}%", "Scale 7+")

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# SHORT INSIGHTS
# ============================================================

rank_df = filtered.dropna(subset=[metric_col]).copy()

best = rank_df.sort_values(metric_col, ascending=False).head(1)
weak = rank_df.sort_values(metric_col, ascending=True).head(1)

st.markdown('<div class="section-title">What to notice</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Three quick signals from the current benchmark.</div>',
    unsafe_allow_html=True
)

i1, i2, i3 = st.columns(3)

if not best.empty:
    r = best.iloc[0]
    i1.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Leading cue</div>
            <div class="insight-copy">
                <span class="pill">{r['norm_grade']}</span><br>
                <b>{r['parameter_name']}</b> leads on <b>{metric_label}</b> at <b>{r[metric_col]:.2f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

if not weak.empty:
    r = weak.iloc[0]
    i2.markdown(
        f"""
        <div class="insight-card">
            <div class="insight-title">Watch-out</div>
            <div class="insight-copy">
                <span class="pill">{r['norm_grade']}</span><br>
                <b>{r['parameter_name']}</b> trails on <b>{metric_label}</b> at <b>{r[metric_col]:.2f}</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

segment_text = selected_slice.replace("_", " ").title()

i3.markdown(
    f"""
    <div class="insight-card">
        <div class="insight-title">Current cut</div>
        <div class="insight-copy">
            <span class="pill">{segment_text}</span><br>
            <b>{len(filtered):,}</b> norm rows · minimum base <b>{min_base}</b>.
        </div>
    </div>
    """,
    unsafe_allow_html=True
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
        base=("base", "sum")
    )
    .reset_index()
)

chart_df = chart_df.sort_values(metric_col, ascending=False).head(top_n * 3)

# ============================================================
# CHARTS
# ============================================================

left, right = st.columns([1.35, 1])

with left:
    st.markdown('<div class="section-title">Attribute ranking</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">Top attributes by {metric_label}.</div>',
        unsafe_allow_html=True
    )

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
            "Bottom 25%": "#C8BFB2"
        }
    )

    fig.update_layout(
        height=540,
        plot_bgcolor=CREAM,
        paper_bgcolor=CREAM,
        font=dict(color=NAVY, size=13),
        legend_title_text="",
        xaxis_title=metric_label,
        yaxis_title="",
        margin=dict(l=0, r=10, t=10, b=10)
    )

    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Norm group read</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-subtitle">Average {metric_label} by group.</div>',
        unsafe_allow_html=True
    )

    grade_df = (
        filtered
        .dropna(subset=[metric_col])
        .groupby("norm_grade", observed=False)
        .agg(value=(metric_col, "mean"), base=("base", "sum"))
        .reset_index()
    )

    fig2 = px.bar(
        grade_df,
        x="norm_grade",
        y="value",
        color="norm_grade",
        text="value",
        color_discrete_map={
            "Top 25%": GOLD,
            "Average 50%": BLUE,
            "Bottom 25%": "#C8BFB2"
        }
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
        margin=dict(l=0, r=10, t=10, b=10)
    )

    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# GAP ANALYSIS
# ============================================================

st.markdown('<div class="section-title">Biggest norm gaps</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Where Top 25% and Bottom 25% differ the most.</div>',
    unsafe_allow_html=True
)

gap_source = filtered[
    filtered["norm_grade"].astype(str).isin(["Top 25%", "Bottom 25%"])
].copy()

gap_pivot = gap_source.pivot_table(
    index=["parameter_name", "scale"],
    columns="norm_grade",
    values=metric_col,
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
        color="gap",
        color_continuous_scale=[[0, "#EDE6DA"], [1, GOLD]],
        hover_data=["scale", "Top 25%", "Bottom 25%"]
    )

    fig3.update_layout(
        height=460,
        plot_bgcolor=CREAM,
        paper_bgcolor=CREAM,
        font=dict(color=NAVY, size=13),
        xaxis_title=f"Gap in {metric_label}",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(l=0, r=10, t=10, b=10)
    )

    fig3.update_yaxes(autorange="reversed")
    st.plotly_chart(fig3, use_container_width=True)

else:
    st.info("Select both Top 25% and Bottom 25% to see the gap view.")

st.markdown("<hr>", unsafe_allow_html=True)

# ============================================================
# CLEAN TABLE
# ============================================================

st.markdown('<div class="section-title">Norm table</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Clean, export-ready benchmark table.</div>',
    unsafe_allow_html=True
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

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True
)

csv = table_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download CSV",
    csv,
    "deka_norm_filtered.csv",
    "text/csv"
)

st.markdown(
    """
    <p class="small-note">
        Norm groups are calculated from ranked respondent-level scores within each parameter and scale.
    </p>
    """,
    unsafe_allow_html=True
)
