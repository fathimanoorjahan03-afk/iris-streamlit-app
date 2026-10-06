import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, rgba(88, 128, 255, 0.18), transparent 30%), #0b1220;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    .stDataFrame, .stTable {
        border-radius: 14px;
    }
    div[data-testid="stMetric"] > div {
        background: rgba(15, 23, 41, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Iris Analysis")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = iris.target_names[iris.target]

st.subheader("Dataset preview")
st.dataframe(df.head())

st.subheader("Summary statistics")
st.dataframe(df.describe().T)

st.subheader("Species distribution")
count_df = df["species"].value_counts().rename_axis("Species").reset_index(name="Count")
st.bar_chart(count_df.set_index("Species"))

st.subheader("Feature distribution by species")
feature_cols = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
for feature in feature_cols:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="species", y=feature, ax=ax, palette="Set2")
    ax.set_title(f"{feature} by species")
    st.pyplot(fig)

st.subheader("Petal length vs petal width")
fig, ax = plt.subplots(figsize=(7, 5))
for species in df["species"].unique():
    subset = df[df["species"] == species]
    ax.scatter(subset["petal length (cm)"], subset["petal width (cm)"], label=species, s=60, alpha=0.8)
ax.set_xlabel("Petal length (cm)")
ax.set_ylabel("Petal width (cm)")
ax.legend(title="Species")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("Sepal length vs sepal width")
fig, ax = plt.subplots(figsize=(7, 5))
for species in df["species"].unique():
    subset = df[df["species"] == species]
    ax.scatter(subset["sepal length (cm)"], subset["sepal width (cm)"], label=species, s=60, alpha=0.8)
ax.set_xlabel("Sepal length (cm)")
ax.set_ylabel("Sepal width (cm)")
ax.legend(title="Species")
ax.grid(True, alpha=0.3)
st.pyplot(fig)

st.subheader("Correlation heatmap")
correlation = df[feature_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
st.pyplot(fig)

st.subheader("Pairplot-like overview")
selected_features = ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
fig = sns.pairplot(df[selected_features + ["species"]], hue="species", height=2.0, aspect=1.0)
st.pyplot(fig)

st.subheader("Key insights")
st.markdown(
    "- Setosa has clearly smaller petal measurements than the other two species.\n"
    "- Petal length and petal width are strongly related and highly useful for species separation.\n"
    "- Versicolor and virginica overlap more in sepal dimensions, but petal dimensions help distinguish them better.\n"
    "- Correlation analysis shows that petal features are the most informative variables for classification."
)
