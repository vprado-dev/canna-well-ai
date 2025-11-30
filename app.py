"""
Cannabis Strain Recommender - Streamlit App
"""

import streamlit as st
import logging

from src.config import (
    MEDICAL_COLS,
    POS_EFFECT_COLS,
    NEG_EFFECT_COLS
)
from src.preprocess import preprocess_data
from src.models import load_models
from src.recommender import (
    build_user_vector,
    recommend_strains_global_knn,
    recommend_strains_cluster_knn
)
from src.utils import get_strain_summary, log_medical_scores

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Cache data and models for fast reloading
@st.cache_data
def load_processed_data():
    """Load and cache the preprocessed strain data."""
    return preprocess_data()


@st.cache_resource
def load_trained_models():
    """Load and cache the trained models."""
    return load_models()


def main():
    """Main Streamlit app."""

    # Page config
    st.set_page_config(
        page_title="Cannabis Strain Recommender",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Custom CSS for green theme enhancements
    st.markdown("""
        <style>
        /* Main title styling */
        h1 {
            color: #2E7D32 !important;
            font-weight: 600 !important;
        }

        /* Subheaders */
        h2, h3 {
            color: #388E3C !important;
        }

        /* Info boxes */
        .stAlert {
            background-color: #E8F5E9 !important;
            border-left: 4px solid #4CAF50 !important;
        }

        /* Success boxes */
        .stSuccess {
            background-color: #C8E6C9 !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #4CAF50 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 500 !important;
            border: none !important;
            box-shadow: 0 2px 4px rgba(76, 175, 80, 0.2) !important;
            transition: all 0.3s ease !important;
        }

        .stButton>button:hover {
            background-color: #45a049 !important;
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3) !important;
            transform: translateY(-1px) !important;
        }

        /* Radio buttons */
        .stRadio > label {
            font-weight: 500 !important;
            color: #2E7D32 !important;
        }

        /* Multiselect */
        .stMultiSelect > label {
            font-weight: 500 !important;
            color: #2E7D32 !important;
        }

        /* Containers for strain cards */
        .element-container {
            border-radius: 8px;
        }

        /* Markdown headers in strain cards */
        h3 {
            padding-top: 10px;
            border-bottom: 2px solid #81C784;
            padding-bottom: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("🌿 Cannabis Strain Recommender")
    st.markdown(
        "Find the perfect cannabis strain based on your medical conditions and desired effects."
    )
    st.markdown("---")

    # Load data and models
    try:
        df_med = load_processed_data()
        models = load_trained_models()

        # Add cluster assignments to DataFrame
        X_medical = df_med[MEDICAL_COLS].values
        X_medical_scaled = models["scaler_kmeans"].transform(X_medical)
        df_med["cluster"] = models["kmeans"].predict(X_medical_scaled)

    except Exception as e:
        st.error(f"Error loading data or models: {e}")
        st.info("Please make sure you have run `python train_models.py` first.")
        st.stop()

    # Recommendation method selection
    st.subheader("🔍 Recommendation Method")
    method = st.radio(
        "Choose how to find recommendations:",
        ["Global KNN (Search all strains)", "Cluster-based KNN (Match by medical profile first)"],
        help="**Global KNN** searches across all 2,921 strains. **Cluster-based KNN** first groups you with similar medical profiles, then searches within that group."
    )

    st.markdown("---")

    # User input form
    st.subheader("📋 Your Profile")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Medical Conditions")
        # Sort medical conditions alphabetically, with common ones at top
        common_conditions = ["anxiety", "stress", "pain", "depression", "insomnia"]
        sorted_medical = common_conditions + sorted([c for c in MEDICAL_COLS if c not in common_conditions])

        diseases = st.multiselect(
            "Select the medical conditions you want to treat:",
            options=sorted_medical,
            help="Choose one or more conditions"
        )

        st.markdown("### Desired Effects")
        desired_effects = st.multiselect(
            "What effects do you want to feel?",
            options=sorted(POS_EFFECT_COLS),
            help="Select positive effects you're looking for"
        )

    with col2:
        st.markdown("### Effects to Avoid")
        avoid_effects = st.multiselect(
            "What effects do you want to avoid?",
            options=sorted(NEG_EFFECT_COLS),
            help="Select negative effects you want to minimize"
        )

        st.markdown("### Number of Recommendations")
        n_recommendations = st.slider(
            "How many strains to recommend?",
            min_value=5,
            max_value=20,
            value=10,
            help="Number of strain recommendations to display"
        )

    st.markdown("---")

    # Get recommendations button
    if st.button("🔎 Get Recommendations", type="primary", use_container_width=True):
        # Validate input
        if not diseases and not desired_effects:
            st.warning("⚠️ Please select at least one medical condition or desired effect.")
            return

        # Build user vector
        user_vector = build_user_vector(
            diseases=diseases,
            desired_effects=desired_effects,
            avoid_effects=avoid_effects
        )

        # Get recommendations based on selected method
        with st.spinner("Finding your perfect strains..."):
            if method.startswith("Global KNN"):
                recommendations = recommend_strains_global_knn(
                    user_vector=user_vector,
                    knn_model=models["knn_model"],
                    scaler_knn=models["scaler_knn"],
                    df=df_med,
                    n_neighbors=n_recommendations
                )
                st.info(f"📊 Searched across all {len(df_med)} strains")
            else:
                recommendations = recommend_strains_cluster_knn(
                    user_vector=user_vector,
                    diseases=diseases if diseases else [],
                    kmeans=models["kmeans"],
                    scaler_kmeans=models["scaler_kmeans"],
                    scaler_knn=models["scaler_knn"],
                    df=df_med,
                    n_neighbors=n_recommendations
                )
                if not recommendations.empty:
                    cluster_id = recommendations.iloc[0]["cluster"]
                    st.info(f"📊 Matched to medical profile cluster {cluster_id}")

        # Display results
        if recommendations.empty:
            st.warning("No recommendations found. Try adjusting your selections.")
            return

        st.markdown("---")
        st.subheader(f"✨ Top {len(recommendations)} Recommendations")

        # Display each recommendation
        for idx, (_, strain_row) in enumerate(recommendations.iterrows(), 1):
            # Get full strain data from df_med
            strain_full = df_med[df_med["name"] == strain_row["name"]].iloc[0].copy()

            # Add the knn_distance from recommendations
            strain_full["knn_distance"] = strain_row["knn_distance"]

            # Get formatted summary
            summary = get_strain_summary(
                strain_full,
                POS_EFFECT_COLS,
                NEG_EFFECT_COLS,
                threshold=10.0
            )

            # Log medical scores (not displayed)
            log_medical_scores(strain_full, MEDICAL_COLS)

            # Display strain card
            with st.container():
                st.markdown(f"### {idx}. {summary['name']}")

                col_a, col_b, col_c = st.columns([2, 2, 1])

                with col_a:
                    st.write(f"**Type:** {summary['type']}")

                with col_b:
                    st.write(f"**THC Level:** {summary['thc_level']}")

                with col_c:
                    st.write(f"**Distance:** {summary['match_score']}")
                    st.caption("(lower = better match)")

                # Positive effects
                if summary['positive_effects']:
                    st.markdown("**✅ Positive Effects:**")
                    st.write(", ".join(summary['positive_effects']))
                else:
                    st.write("**✅ Positive Effects:** None above threshold")

                # Negative effects
                if summary['negative_effects']:
                    st.markdown("**⚠️ Negative Effects:**")
                    st.write(", ".join(summary['negative_effects']))
                else:
                    st.write("**⚠️ Negative Effects:** None above threshold")

                st.markdown("---")

    # Footer
    st.markdown("")
    st.markdown("")
    with st.expander("ℹ️ About this App"):
        st.markdown("""
        This app uses machine learning to recommend cannabis strains based on your needs:

        - **K-Means Clustering**: Groups strains by medical effectiveness profiles
        - **K-Nearest Neighbors (KNN)**: Finds strains most similar to your preferences

        **Data**: 2,921 strains from Leafly with 39 medical conditions, 13 positive effects, and 6 negative effects.

        **Distance Score**: This is the Euclidean distance between your profile and each strain in the 58-dimensional feature space.
        - **Lower values = better matches** (typically 5-15 for good matches)
        - Not a percentage - it's a similarity metric
        - The top recommendation has the smallest distance to your ideal profile
        """)


if __name__ == "__main__":
    main()
