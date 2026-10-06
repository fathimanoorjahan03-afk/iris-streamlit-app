import pickle
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

st.title("🌿 Iris Species Prediction Studio")
st.caption("Adjust the flower measurements and click Predict to see the result.")

st.markdown(
    """
    <style>
    .prediction-card {
        background: linear-gradient(135deg, rgba(18, 118, 95, 0.24), rgba(71, 102, 255, 0.14));
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 1.2rem 1.2rem;
        margin: 0.8rem 0 1.4rem 0;
        box-shadow: 0 6px 16px rgba(0,0,0,0.18);
    }
    .section-title {
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
        font-size: 1.2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()
iris = load_iris(as_frame=True)
class_names = iris.target_names
feature_names = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

species_means = (
    iris.frame.copy()
    .assign(species=class_names[iris.target])
    .groupby("species")[feature_names]
    .mean()
)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.5, step=0.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, step=0.1)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, step=0.1)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.2, step=0.1)

predict_button = st.button("Predict species", type="primary")

if predict_button:
    values = {
        "sepal length (cm)": sepal_length,
        "sepal width (cm)": sepal_width,
        "petal length (cm)": petal_length,
        "petal width (cm)": petal_width,
    }

    row = np.array([values[name] for name in feature_names], dtype=float).reshape(1, -1)
    prediction_index = model.predict(row)[0]
    prediction = class_names[prediction_index]
    probability = model.predict_proba(row)[0]
    confidence = probability.max() * 100

    st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Prediction result")
        st.markdown(f"### 🏷️ Predicted species: <span style='color:#7ee7c3; font-weight:700'>{prediction}</span>", unsafe_allow_html=True)
    with col2:
        st.metric("Confidence", f"{confidence:.1f}%")
    st.markdown("</div>", unsafe_allow_html=True)

    prob_df = pd.DataFrame({
        "Species": class_names,
        "Probability": probability,
    }).set_index("Species")

    st.markdown('<div class="section-title">Probability distribution</div>', unsafe_allow_html=True)
    st.bar_chart(prob_df)

    comparison_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Current input": [values[f] for f in feature_names],
            "Setosa": species_means.loc["setosa"].values,
            "Versicolor": species_means.loc["versicolor"].values,
            "Virginica": species_means.loc["virginica"].values,
        }
    ).set_index("Feature")

    st.markdown('<div class="section-title">Feature comparison</div>', unsafe_allow_html=True)
    st.line_chart(comparison_df)

    st.markdown('<div class="section-title">Feature breakdown</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(feature_names))
    ax.bar(x, [values[f] for f in feature_names], width=0.7, label="Current input", color="#5EA1FF")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=20, ha="right")
    ax.set_ylabel("Value (cm)")
    ax.set_title("Current feature values")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.patch.set_facecolor("#0f1729")
    ax.set_facecolor("#0f1729")
    ax.tick_params(axis='x', colors='white', labelsize=10)
    ax.tick_params(axis='y', colors='white', labelsize=10)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    legend = ax.legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('white')

    for spine in ax.spines.values():
        spine.set_color("#334155")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown('<div class="section-title">Decision insight</div>', unsafe_allow_html=True)
    st.markdown(
        f"The model predicts this sample is most likely a **{prediction}** flower with **{confidence:.1f}%** confidence. "
        "The feature comparison chart shows how the current values compare with the average values for each species in the Iris dataset.",
        unsafe_allow_html=True,
    )
