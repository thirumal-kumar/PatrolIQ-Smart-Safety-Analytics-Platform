##############################################
# PatrolIQ — Streamlit Cloud Version (Option B)
# Loads data from Google Drive ON DEMAND
# Same UI & Pages as local version, cloud-optimized
##############################################

import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# -----------------------
# Theme colors
# -----------------------
NAVY = "#0A1B2A"
BLUE = "#1F6FEB"

st.set_page_config(layout="wide", page_title="PatrolIQ — Smart Safety Dashboard")

# -----------------------
# Google Drive URLs (Direct Download)
# -----------------------
URL_CLUSTERS = "https://drive.google.com/u/0/uc?id=1F4UDJJ7F3IhDilADMWVKLkB9I8k2iU65&export=download"
URL_EMBEDDINGS = "https://drive.google.com/u/0/uc?id=1J7NhBreqtrY99iFlmgvf-D-qHFqFhHW0&export=download"
URL_FEATURES = "https://drive.google.com/u/0/uc?id=1oUgVL6Ns_NlnAii_DUgMfFuasLbJj3jL&export=download"

# -----------------------
# Load parquet from Google Drive
# -----------------------
@st.cache_data(show_spinner=True)
def load_parquet_drive(url):
    r = requests.get(url)
    r.raise_for_status()
    return pd.read_parquet(BytesIO(r.content))


# -----------------------
# Sidebar Controls
# -----------------------
st.sidebar.header("Controls — PatrolIQ")

CLUSTER_MAP = {
    "K-Means Clusters": "kmeans",
    "Hierarchical Clusters": "hierarchical",
    "DBSCAN Clusters": "dbscan_tile"
}

cluster_choice_label = st.sidebar.selectbox("Cluster Layer", list(CLUSTER_MAP.keys()))
cluster_col = CLUSTER_MAP[cluster_choice_label]

sample_map = st.sidebar.slider("Map sample size", 10000, 100000, 50000, step=5000)

page = st.sidebar.radio("Navigate", [
    "Home",
    "Map / Hotspots",
    "Temporal Analysis",
    "Cluster Comparison",
    "PCA & UMAP",
    "Model Performance",
    "Patrol Recommendations"
])


##############################################
# PAGE 1 — HOME
##############################################
if page == "Home":
    st.markdown(f"<h2 style='color:{NAVY};'>PatrolIQ — Overview</h2>", unsafe_allow_html=True)

    st.markdown("""
    This dashboard uses Chicago crime data to visualize:
    - Geographic hotspots  
    - Temporal crime patterns  
    - Clustering insights  
    - Patrol recommendations  
    """)

    df = load_parquet_drive(URL_CLUSTERS)

    st.metric("Total Records", f"{len(df):,}")
    if "primary_type" in df.columns:
        st.metric("Unique Crime Types", df["primary_type"].nunique())

    st.subheader("Top Crime Types")
    if "primary_type" in df.columns:
        top = df["primary_type"].value_counts().head(8)
        fig = px.bar(
            x=top.index, 
            y=top.values,
            labels={"x": "Crime Type", "y": "Count"},
            color=top.index,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        st.plotly_chart(fig, use_container_width=True)


##############################################
# PAGE 2 — MAP / HOTSPOTS
##############################################
elif page == "Map / Hotspots":
    st.markdown(f"<h2 style='color:{NAVY};'>Geographic Hotspots</h2>", unsafe_allow_html=True)

    df = load_parquet_drive(URL_CLUSTERS).dropna(subset=["latitude", "longitude"])
    df = df.sample(min(sample_map, len(df)))

    if cluster_col not in df.columns:
        st.error(f"Column '{cluster_col}' not found.")
        st.stop()

    # Safe color generation
    codes = df[cluster_col].astype("category").cat.codes.astype(np.int32)
    df["r"] = (codes * 13) % 255
    df["g"] = (codes * 7) % 255
    df["b"] = (codes * 3) % 255

    view = pdk.ViewState(
        latitude=float(df["latitude"].median()),
        longitude=float(df["longitude"].median()),
        zoom=10
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[longitude, latitude]',
        get_color='[r, g, b]',
        get_radius=50,
        pickable=True,
        auto_highlight=True
    )

    tooltip = {
        "html": "<b>Crime:</b> {primary_type}<br><b>Date:</b> {date}<br><b>Cluster:</b> {"
                + cluster_col + "}",
        "style": {"backgroundColor": NAVY, "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip))


##############################################
# PAGE 3 — TEMPORAL ANALYSIS
##############################################
elif page == "Temporal Analysis":
    st.markdown(f"<h2 style='color:{NAVY};'>Temporal Crime Patterns</h2>", unsafe_allow_html=True)
    df = load_parquet_drive(URL_CLUSTERS)

    if "hour" in df.columns and "day_of_week" in df.columns:
        pivot = df.pivot_table(index="hour", columns="day_of_week",
                               values="id", aggfunc="count", fill_value=0)
        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=list(range(7)),
            y=pivot.index,
            colorscale="YlOrRd"
        ))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Hour or day_of_week missing.")


##############################################
# PAGE 4 — CLUSTER COMPARISON
##############################################
elif page == "Cluster Comparison":
    st.markdown(f"<h2 style='color:{NAVY};'>Cluster Comparison</h2>", unsafe_allow_html=True)
    df = load_parquet_drive(URL_CLUSTERS)

    if cluster_col not in df.columns:
        st.error("Cluster column missing.")
        st.stop()

    cluster_list = df[cluster_col].value_counts().index.tolist()
    cid = st.selectbox("Select cluster", cluster_list)

    cdf = df[df[cluster_col] == cid]

    st.metric("Records", len(cdf))

    if "primary_type" in cdf.columns:
        top = cdf["primary_type"].value_counts().head(10)
        fig = px.bar(top, labels={"index": "Crime Type", "value": "Count"})
        st.plotly_chart(fig, use_container_width=True)


##############################################
# PAGE 5 — PCA & UMAP
##############################################
elif page == "PCA & UMAP":
    st.markdown(f"<h2 style='color:{NAVY};'>PCA & UMAP</h2>", unsafe_allow_html=True)
    df = load_parquet_drive(URL_EMBEDDINGS)

    if "pca1" in df.columns and "pca2" in df.columns:
        st.subheader("PCA Scatter")
        fig = px.scatter(df.sample(20000),
                         x="pca1", y="pca2",
                         color=df[cluster_col].astype(str)
                         if cluster_col in df.columns else None)
        st.plotly_chart(fig, use_container_width=True)

    if "umap1" in df.columns and "umap2" in df.columns:
        st.subheader("UMAP Scatter")
        fig = px.scatter(df.sample(20000),
                         x="umap1", y="umap2",
                         color=df[cluster_col].astype(str)
                         if cluster_col in df.columns else None)
        st.plotly_chart(fig, use_container_width=True)


##############################################
# PAGE 6 — MODEL PERFORMANCE
##############################################
elif page == "Model Performance":
    st.markdown(f"<h2 style='color:{NAVY};'>Cluster Model Performance</h2>", unsafe_allow_html=True)
    df = load_parquet_drive(URL_CLUSTERS)

    rows = []
    for col in ["kmeans", "hierarchical", "dbscan_tile"]:
        if col not in df.columns:
            continue
        sample = df.sample(min(20000, len(df)))
        X = sample[["latitude", "longitude"]].values
        y = sample[col].values
        try:
            sil = round(silhouette_score(X, y), 3)
        except:
            sil = "N/A"
        rows.append({"method": col, "silhouette": sil})

    st.dataframe(pd.DataFrame(rows))


##############################################
# PAGE 7 — PATROL RECOMMENDATIONS
##############################################
elif page == "Patrol Recommendations":
    st.markdown(f"<h2 style='color:{NAVY};'>Patrol Recommendations</h2>", unsafe_allow_html=True)
    df = load_parquet_drive(URL_CLUSTERS)

    if "kmeans" not in df.columns:
        st.error("KMeans cluster not found.")
        st.stop()

    summary = df.groupby("kmeans").agg(
        count=("id", "count"),
        severity=("crime_severity", "mean")
    ).reset_index()

    summary["score"] = summary["count"] * summary["severity"].fillna(1)
    summary = summary.sort_values("score", ascending=False)

    st.subheader("Top Clusters")
    st.dataframe(summary.head(10))


