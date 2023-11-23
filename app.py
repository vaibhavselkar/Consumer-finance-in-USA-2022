import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Load data
df = pd.read_csv("SCFP2022.csv")
mask = df["TURNFEAR"] == 1
df = df[mask]

# Function to get high variance features
def get_high_var_features(trimmed=True, return_feat_names=True):
    if trimmed:
        top_five_features = df.apply(trimmed_var).sort_values().tail(5)
    else:
        top_five_features = df.var().sort_values().tail(5)
    if return_feat_names:
        top_five_features = top_five_features.index.tolist()
    return top_five_features

# Function to get model metrics
def get_model_metrics(trimmed=True, k=2, return_metrics=False):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=42))
    model.fit(X)
    if return_metrics:
        i = model.named_steps["kmeans"].inertia_
        ss = silhouette_score(X, model.named_steps["kmeans"].labels_)
        metrics = {
            "inertia": round(i),
            "silhouette": round(ss, 3)
        }
        return metrics
    return model

# Function to get PCA labels
def get_pca_labels(trimmed=True, k=2):
    features = get_high_var_features(trimmed=trimmed, return_feat_names=True)
    X = df[features]
    transformer = PCA(n_components=2, random_state=42)
    X_t = transformer.fit_transform(X)
    X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])
    model = get_model_metrics(trimmed=trimmed, k=k, return_metrics=False)
    X_pca["labels"] = model.named_steps["kmeans"].labels_.astype(str)
    X_pca.sort_values("labels", inplace=True)
    return X_pca

# Streamlit app
st.title("Survey of Consumer Finances - 2022")

# Button to show the user manual
if st.button("Show User Manual"):
    st.markdown("""
    ## User Manual: USA Finance Data App - 2022

    **1. Introduction**
    The USA Finance Data App is a user-friendly tool designed to provide insights into consumer finance data. This app utilizes the Survey of Consumer Finances (SCF) dataset to analyze high-variance features, perform K-means clustering, and visualize results through interactive charts.

    **2. Navigation**
    **2.1. Variance Analysis**
    You will find a horizontal bar chart displaying the five highest-variance features in the SCF dataset.
    Use the radio buttons to choose between "trimmed" and "not trimmed" variance calculation.
                
    **2.2. K-means Clustering**
    Move the slider to select the number of clusters (k) for K-means clustering.
    Metrics such as inertia and silhouette score will be displayed to evaluate clustering performance.
                
    **2.3. PCA Scatter Plot**
    The app provides a PCA representation of the clusters in a scatter plot.
    Observe how data points are distributed in two principal components.

    **3. How to Use**
    **3.1. Variance Analysis**
    Choose the variance calculation method (trimmed or not trimmed) using the radio buttons.
    Explore the bar chart to identify high-variance features in the dataset.
                
    **3.2. K-means Clustering**
    Adjust the slider to set the number of clusters (k) for K-means clustering.
    Metrics will update dynamically based on the selected number of clusters.
                
    **3.3. PCA Scatter Plot**
    Visualize the clusters in the scatter plot on the app.
    Explore how data points are grouped based on K-means clustering.

    **4. Additional Information**
    This app is built using Streamlit, a user-friendly Python library for creating interactive web applications.
    The underlying data is sourced from the Survey of Consumer Finances (SCF) 2022 dataset.
    For more information on the SCF dataset, visit [Federal Reserve SCF](https://www.federalreserve.gov/econres/scfindex.htm).
    """)

st.header("High Variance Features")

# Bar chart element
trimmed_option = st.radio("Variance Calculation", ["trimmed", "not trimmed"], index=0)
trimmed = True if trimmed_option == "trimmed" else False

top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
fig = px.bar(x=top_five_features, y=top_five_features.index, orientation="h")
fig.update_layout(xaxis_title="Variance", yaxis_title="Features")
st.write(fig)

st.header("K-means clustering")
k = st.slider("Number of Clusters (k)", min_value=2, max_value=12, step=1, value=2)

metrics = get_model_metrics(trimmed=trimmed, k=k, return_metrics=True)
st.subheader("Model Metrics")
st.write(f"Inertia: {metrics['inertia']}")
st.write(f"Silhouette Score: {metrics['silhouette']}")

# PCA Scatter plot
fig = px.scatter(
    data_frame=get_pca_labels(trimmed=trimmed, k=k),
    x="PC1",
    y="PC2",
    color="labels",
    title="PCA Representation of Clusters"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
st.plotly_chart(fig)