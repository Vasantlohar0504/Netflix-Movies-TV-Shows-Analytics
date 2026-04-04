import sys
import os
import pycountry
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_cleaning import load_dataset, clean_dataset, get_genre_counts

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(page_title="Netflix Analytics Dashboard", layout="wide")

# ------------------------------------------------
# DASHBOARD STYLE
# ------------------------------------------------

st.markdown("""
<style>
.main {background-color:#0e1117;}
h1,h2,h3{color:white;}
[data-testid="stMetricValue"]{
font-size:28px;
font-weight:bold;
color:#ff4b4b;
}
</style>
""", unsafe_allow_html=True)

st.title("🎬 Netflix Movies & TV Shows Analytics Dashboard")

# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

df = load_dataset("dataset/netflix_titles.csv")
df = clean_dataset(df)

df["popularity_score"] = (
    df["release_year"]*0.3 +
    df["duration_int"].fillna(0)*0.2
)

# ------------------------------------------------
# SIDEBAR FILTERS
# ------------------------------------------------

st.sidebar.header("Dashboard Filters")

type_filter = st.sidebar.multiselect(
"Content Type", df["type"].unique(), default=df["type"].unique()
)

country_filter = st.sidebar.multiselect(
"Country", df["country"].dropna().unique()
)

rating_filter = st.sidebar.multiselect(
"Rating", df["rating"].dropna().unique()
)

genre_filter = st.sidebar.multiselect(
"Genre", df["listed_in"].str.split(", ").explode().unique()
)

year_filter = st.sidebar.slider(
"Release Year",
int(df["release_year"].min()),
int(df["release_year"].max()),
(2000,2021)
)

# ------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------

df_filtered = df[df["type"].isin(type_filter)]

df_filtered = df_filtered[
(df_filtered["release_year"]>=year_filter[0]) &
(df_filtered["release_year"]<=year_filter[1])
]

if country_filter:
    df_filtered = df_filtered[df_filtered["country"].isin(country_filter)]

if rating_filter:
    df_filtered = df_filtered[df_filtered["rating"].isin(rating_filter)]

if genre_filter:
    df_filtered = df_filtered[
        df_filtered["listed_in"].str.contains("|".join(genre_filter))
    ]

# ------------------------------------------------
# KPI METRICS
# ------------------------------------------------

st.subheader("Dataset Overview")

col1,col2,col3,col4 = st.columns(4)

col1.metric("Total Titles",len(df_filtered))
col2.metric("Movies",(df_filtered["type"]=="Movie").sum())
col3.metric("TV Shows",(df_filtered["type"]=="TV Show").sum())
col4.metric("Countries",df_filtered["country"].nunique())

# ------------------------------------------------
# TABS
# ------------------------------------------------

tab1,tab2,tab3,tab4,tab5,tab6,tab7,tab8 = st.tabs([
"📈 Trends",
"🌍 Countries",
"🎭 Genres",
"⭐ Cast & Crew",
"📊 Clustering",
"🤖 Recommendation",
"📊 Strategy Insights",
"📄 Dataset"
])

# ------------------------------------------------
# TAB 1 — TRENDS
# ------------------------------------------------

with tab1:

    year_data = df_filtered["release_year"].value_counts().sort_index()

    fig1 = px.line(
        x=year_data.index,
        y=year_data.values,
        title="Netflix Content Growth",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig1,width="stretch")

    fig2 = px.bar(
        x=year_data.index,
        y=year_data.values,
        title="Content Added Per Year",
        color=year_data.values,
        color_continuous_scale="viridis"
    )
    st.plotly_chart(fig2,width="stretch")

    type_counts = df_filtered["type"].value_counts()

    fig3 = px.pie(
        values=type_counts.values,
        names=type_counts.index,
        title="Movies vs TV Shows",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig3,width="stretch")

# ------------------------------------------------
# TAB 2 — COUNTRIES
# ------------------------------------------------

with tab2:

    st.subheader("🌍 Country Content Analysis")

    country_counts = df_filtered["country"].value_counts().head(10)

    # ---------------- BAR CHART ----------------
    fig4 = px.bar(
        x=country_counts.values,
        y=country_counts.index,
        orientation="h",
        title="Top Countries Producing Content",
        color=country_counts.values,
        color_continuous_scale="Reds"
    )

    st.plotly_chart(fig4, width="stretch")


    # ---------------- PIE CHART ----------------
    fig5 = px.pie(
        values=country_counts.values,
        names=country_counts.index,
        title="Country Share",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )

    st.plotly_chart(fig5, width="stretch")


    # ---------------- HISTOGRAM ----------------
    fig6 = px.histogram(
        df_filtered,
        x="country",
        title="Country Distribution",
        color_discrete_sequence=["orange"]
    )

    st.plotly_chart(fig6, width="stretch")


    # ---------------- WORLD MAP ----------------

    st.subheader("🌎 Global Netflix Content Map")

    country_map = (
        df_filtered["country"]
        .value_counts()
        .reset_index()
    )

    country_map.columns = ["country", "count"]

    # Convert country names to ISO3 codes
    def get_iso3(country_name):
        try:
            return pycountry.countries.lookup(country_name).alpha_3
        except:
            return None

    country_map["iso3"] = country_map["country"].apply(get_iso3)

    country_map = country_map.dropna(subset=["iso3"])

    fig_map = px.choropleth(
        country_map,
        locations="iso3",
        color="count",
        hover_name="country",
        color_continuous_scale="Reds",
        title="Netflix Content Distribution Worldwide"
    )

    fig_map.update_layout(
        height=700,
        margin=dict(l=0, r=0, t=50, b=0)
    )

    st.plotly_chart(fig_map, width="stretch")
    
# ------------------------------------------------
# TAB 3 — GENRES
# ------------------------------------------------

with tab3:

    genres = get_genre_counts(df_filtered).head(10)

    fig7 = px.bar(
        x=genres.values,
        y=genres.index,
        orientation="h",
        title="Top Genres",
        color=genres.values,
        color_continuous_scale="Purples"
    )
    st.plotly_chart(fig7,width="stretch")

    genre_df = (
        df_filtered
        .assign(genre=df_filtered["listed_in"].str.split(", "))
        .explode("genre")
    )

    genre_trend = (
        genre_df.groupby(["release_year","genre"])
        .size()
        .reset_index(name="count")
    )

    fig8 = px.line(
        genre_trend,
        x="release_year",
        y="count",
        color="genre",
        title="Genre Evolution",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    st.plotly_chart(fig8,width="stretch")

    fig9 = px.histogram(
        genre_df,
        x="genre",
        title="Genre Distribution",
        color_discrete_sequence=["green"]
    )
    st.plotly_chart(fig9,width="stretch")

# ------------------------------------------------
# TAB 4 — CAST & CREW
# ------------------------------------------------

with tab4:

    directors = (
        df_filtered["director"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    fig10 = px.bar(
        x=directors.values,
        y=directors.index,
        orientation="h",
        title="Top Directors",
        color_discrete_sequence=["#ff7f0e"]
    )
    st.plotly_chart(fig10,width="stretch")

    actors = (
        df_filtered["cast"]
        .dropna()
        .str.split(", ")
        .explode()
        .value_counts()
        .head(10)
    )

    fig11 = px.bar(
        x=actors.values,
        y=actors.index,
        orientation="h",
        title="Top Actors",
        color_discrete_sequence=["#2ca02c"]
    )
    st.plotly_chart(fig11,width="stretch")

    fig12 = px.histogram(
        df_filtered,
        x="duration_int",
        title="Content Duration Distribution",
        color_discrete_sequence=["#9467bd"]
    )
    st.plotly_chart(fig12,width="stretch")

# ------------------------------------------------
# TAB 5 — CLUSTERING
# ------------------------------------------------

with tab5:

    st.subheader("Content Clustering")

    cluster_df = df_filtered[["title","listed_in","description"]].dropna()

    if len(cluster_df) < 2:

        st.warning("Not enough data for clustering. Please adjust filters.")

    else:

        text_data = cluster_df["listed_in"] + " " + cluster_df["description"]

        vectorizer = TfidfVectorizer(stop_words="english")

        X = vectorizer.fit_transform(text_data)

        # Dynamic cluster count
        n_clusters = min(5, len(cluster_df))

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        cluster_df["cluster"] = kmeans.fit_predict(X)

        fig13 = px.scatter(
            x=np.arange(len(cluster_df)),
            y=cluster_df["cluster"],
            hover_name=cluster_df["title"],
            color=cluster_df["cluster"],
            title="Content Clusters",
            color_continuous_scale="Turbo"
        )

        st.plotly_chart(fig13, width="stretch")

        fig14 = px.histogram(
            cluster_df,
            x="cluster",
            title="Cluster Distribution",
            color="cluster",
            color_discrete_sequence=px.colors.qualitative.Set3
        )

        st.plotly_chart(fig14, width="stretch")

        fig15 = px.box(
            cluster_df,
            y="cluster",
            title="Cluster Spread",
            color="cluster",
            color_discrete_sequence=px.colors.qualitative.Bold
        )

        st.plotly_chart(fig15, width="stretch")

# ------------------------------------------------
# TAB 6 — RECOMMENDATION
# ------------------------------------------------

with tab6:

    rec_df = df[["title","description"]].dropna()

    tfidf = TfidfVectorizer(stop_words="english")

    tfidf_matrix = tfidf.fit_transform(rec_df["description"])

    similarity = cosine_similarity(tfidf_matrix)

    title = st.selectbox("Select Title",rec_df["title"])

    idx = rec_df[rec_df["title"]==title].index[0]

    scores = list(enumerate(similarity[idx]))

    scores = sorted(scores,key=lambda x:x[1],reverse=True)[1:6]

    recommendations = [rec_df.iloc[i[0]]["title"] for i in scores]

    st.write("Recommended Titles")

    for r in recommendations:
        st.write("•",r)

# ------------------------------------------------
# TAB 7 — STRATEGY INSIGHTS
# ------------------------------------------------

with tab7:

    top_popular = df_filtered.sort_values(
        "popularity_score",
        ascending=False
    ).head(10)

    fig16 = px.bar(
        top_popular,
        x="popularity_score",
        y="title",
        orientation="h",
        title="Top Popular Titles",
        color="popularity_score",
        color_continuous_scale="Plasma"
    )
    st.plotly_chart(fig16,width="stretch")

    fig17 = px.box(
        df_filtered,
        x="type",
        y="duration_int",
        title="Duration Strategy",
        color="type",
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    st.plotly_chart(fig17,width="stretch")

    growth = df_filtered.groupby("release_year").size()

    model = LinearRegression()

    X = np.array(growth.index).reshape(-1,1)
    y = growth.values

    model.fit(X,y)

    future = np.arange(growth.index.min(),growth.index.max()+5).reshape(-1,1)

    forecast = model.predict(future)

    fig18 = px.line(
        x=future.flatten(),
        y=forecast,
        title="Content Forecast (ML)",
        color_discrete_sequence=["red"]
    )
    st.plotly_chart(fig18,width="stretch")

# ------------------------------------------------
# TAB 8 — DATASET
# ------------------------------------------------

with tab8:

    search = st.text_input("Search Title")

    if search:
        df_filtered = df_filtered[
            df_filtered["title"].str.contains(search,case=False)
        ]

    st.dataframe(df_filtered,width="stretch")

    st.download_button(
        "Download Filtered Data",
        df_filtered.to_csv(index=False),
        "netflix_filtered_data.csv"
    )