import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus

st.set_page_config(page_title="Deka Norm Dashboard", layout="wide")

st.title("📊 Deka Norm Dashboard")

@st.cache_resource
def get_engine():
    user = st.secrets["postgres"]["user"]
    password = quote_plus(st.secrets["postgres"]["password"])
    host = st.secrets["postgres"]["host"]
    port = st.secrets["postgres"]["port"]
    database = st.secrets["postgres"]["database"]

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}?sslmode=require"
    return create_engine(url)

@st.cache_data
def load_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM norm_value_all", engine)

df = load_data()

st.sidebar.header("Filters")

slice_type = st.sidebar.selectbox(
    "Slice Type",
    sorted(df["slice_type"].dropna().unique())
)

filtered = df[df["slice_type"] == slice_type].copy()

parameter = st.sidebar.multiselect(
    "Parameter",
    sorted(filtered["parameter_name"].dropna().unique())
)

if parameter:
    filtered = filtered[filtered["parameter_name"].isin(parameter)]

scale = st.sidebar.multiselect(
    "Scale",
    sorted(filtered["scale"].dropna().unique()),
    default=sorted(filtered["scale"].dropna().unique())
)

if scale:
    filtered = filtered[filtered["scale"].isin(scale)]

norm_grade = st.sidebar.multiselect(
    "Norm Grade",
    sorted(filtered["norm_grade"].dropna().unique()),
    default=sorted(filtered["norm_grade"].dropna().unique())
)

if norm_grade:
    filtered = filtered[filtered["norm_grade"].isin(norm_grade)]

min_base = st.sidebar.number_input("Minimum Base", min_value=0, value=10)

filtered = filtered[filtered["base"] >= min_base]

st.subheader("Summary")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric("Rows", f"{len(filtered):,}")
c2.metric("Avg Mean Score", round(filtered["mean_score"].mean(), 2) if len(filtered) else "-")
c3.metric("Avg TB%", round(filtered["tb_pct"].mean(), 2) if len(filtered) else "-")
c4.metric("Avg T2B%", round(filtered["t2b_pct"].mean(), 2) if len(filtered) else "-")
c5.metric("Total Base", f"{filtered['base'].sum():,}" if len(filtered) else "-")

st.subheader("Norm Value Table")
st.dataframe(filtered, use_container_width=True, hide_index=True)

csv = filtered.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Filtered CSV",
    csv,
    "filtered_norm_value.csv",
    "text/csv"
)
