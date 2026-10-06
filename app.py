import streamlit as st
from sklearn.datasets import load_iris

st.set_page_config(page_title="Iris App", page_icon="🌼", layout="wide")

iris = load_iris(as_frame=True)
df = iris.frame.copy()
df["species"] = iris.target_names[iris.target]

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top left, rgba(81, 119, 255, 0.2), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(31, 211, 170, 0.12), transparent 28%),
                    #0b1220;
    }
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 41, 0.9);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    .big-title {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.12rem;
        color: #d8e5ff;
        margin-bottom: 1rem;
    }
    .info-box {
        background: linear-gradient(135deg, rgba(88, 122, 255, 0.18), rgba(28, 188, 166, 0.10));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1rem 1.2rem;
        margin-top: 0.7rem;
        box-shadow: 0 8px 18px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetric"] > div {
        background: rgba(15, 23, 41, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 14px;
        padding: 0.8rem 1rem;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    p, li, .stMarkdown {
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🌼 Iris Dataset Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A classic classification dataset for learning ML, EDA, and predictive modeling.</div>',
    unsafe_allow_html=True,
)

st.markdown("")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
col2.metric("Features", df.shape[1] - 1)
col3.metric("Species", df["species"].nunique())

st.markdown("")

st.markdown(
    """
    <div class="info-box">
    The Iris flower dataset is one of the most famous datasets in machine learning. It contains measurements of<br>
    three flower species: <b>Setosa</b>, <b>Versicolor</b>, and <b>Virginica</b>.<br>
    It is widely used for learning classification, visualization, and exploratory data analysis.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🌿 Species overview")
species_cols = st.columns(3)
for i, species in enumerate(iris.target_names):
    with species_cols[i]:
        count = (df["species"] == species).sum()
        st.markdown(
            f"""
            <div class="info-box">
                <b>{species.title()}</b><br>
                <span style='font-size:1.3rem;'>{count}</span> samples
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### 📊 Dataset preview")
st.dataframe(df.head(), use_container_width=True)

st.markdown("### 🔍 Features included")
feature_names = [
    "Sepal length (cm)",
    "Sepal width (cm)",
    "Petal length (cm)",
    "Petal width (cm)",
]
for feature in feature_names:
    st.write(f"- {feature}")

st.markdown("### 🧠 Why this dataset matters")
st.write(
    "Iris is perfect for beginners because it is small, clean, and highly interpretable. "
    "It helps us visualize how feature values can separate different classes and how prediction models make decisions."
)

