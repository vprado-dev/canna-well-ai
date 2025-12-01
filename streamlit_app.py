"""
Cannabis Strain Recommender - Streamlit App
"""

import streamlit as st
import pandas as pd
import logging

from src.config import (
    MEDICAL_COLS,
    POS_EFFECT_COLS,
    NEG_EFFECT_COLS,
    AVAILABLE_LOCALES,
    DEFAULT_LOCALE,
    LOCALE_NAMES
)
from src.i18n import init_i18n, t, set_locale, get_current_locale
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
    import os
    from src.i18n import t

    # Check if models exist, if not train them
    if not os.path.exists("models/kmeans_model.pkl"):
        st.info(f"🔄 {t('messages.first_time_setup')}")

        # Import training function
        from src.models import train_and_save_models

        # Load and preprocess data
        df = preprocess_data()

        # Train and save models
        train_and_save_models(df)

        st.success(f"✅ {t('messages.models_trained')}")
        st.balloons()

    return load_models()


def main():
    """Main Streamlit app."""

    # Initialize i18n system (must be before set_page_config to get translations)
    init_i18n(default_locale=DEFAULT_LOCALE)

    # Page config MUST be first Streamlit command
    st.set_page_config(
        page_title=t("app.title"),
        page_icon=t("app.page_icon"),
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Language selector at top (after page config)
    st.markdown(f"### {t('language.selector_label')}")
    col_lang1, col_lang2, col_lang3 = st.columns([1, 2, 3])
    with col_lang1:
        current_locale = get_current_locale()
        selected_locale = st.selectbox(
            "Select language",
            options=AVAILABLE_LOCALES,
            format_func=lambda x: LOCALE_NAMES[x],
            index=AVAILABLE_LOCALES.index(current_locale),
            key="locale_selector",
            label_visibility="collapsed"
        )
        if selected_locale != current_locale:
            set_locale(selected_locale)
            st.rerun()

    st.markdown("---")

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
    st.title(f"🌿 {t('app.title')}")
    st.markdown(t("app.subtitle"))
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
        st.error(t("messages.error_loading", error=str(e)))
        st.info(t("messages.run_training"))
        st.stop()

    # Recommendation method selection
    st.subheader(f"🔍 {t('sections.recommendation_method')}")
    method = st.radio(
        t("method.choose_label"),
        [t("method.global_knn"), t("method.cluster_knn")],
        help=t("help.method_help")
    )

    st.markdown("---")

    # User input form
    st.subheader(f"📋 {t('sections.your_profile')}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### {t('sections.medical_conditions')}")
        # Sort medical conditions alphabetically, with common ones at top
        common_conditions = ["anxiety", "stress", "pain", "depression", "insomnia"]
        sorted_medical = common_conditions + sorted([c for c in MEDICAL_COLS if c not in common_conditions])

        diseases = st.multiselect(
            t("form.conditions_label"),
            options=sorted_medical,
            format_func=lambda x: t(f"medical_conditions.{x}"),
            help=t("help.conditions_help")
        )

        st.markdown(f"### {t('sections.desired_effects')}")
        desired_effects = st.multiselect(
            t("form.desired_label"),
            options=sorted(POS_EFFECT_COLS),
            format_func=lambda x: t(f"effects.positive.{x}"),
            help=t("help.desired_help")
        )

    with col2:
        st.markdown(f"### {t('sections.effects_to_avoid')}")
        avoid_effects = st.multiselect(
            t("form.avoid_label"),
            options=sorted(NEG_EFFECT_COLS),
            format_func=lambda x: t(f"effects.negative.{x}"),
            help=t("help.avoid_help")
        )

        st.markdown(f"### {t('sections.num_recommendations')}")
        n_recommendations = st.slider(
            t("form.num_label"),
            min_value=5,
            max_value=20,
            value=10,
            help=t("help.num_help")
        )

    st.markdown("---")

    # Get recommendations button
    if st.button(f"🔎 {t('buttons.get_recommendations')}", type="primary", use_container_width=True):
        # Validate input
        if not diseases and not desired_effects:
            st.warning(f"⚠️ {t('messages.no_selection')}")
            return

        # Build user vector
        user_vector = build_user_vector(
            diseases=diseases,
            desired_effects=desired_effects,
            avoid_effects=avoid_effects
        )

        # Get recommendations based on selected method
        with st.spinner(t("messages.finding_strains")):
            if method == t("method.global_knn"):
                recommendations = recommend_strains_global_knn(
                    user_vector=user_vector,
                    knn_model=models["knn_model"],
                    scaler_knn=models["scaler_knn"],
                    df=df_med,
                    n_neighbors=n_recommendations
                )
                st.info(f"📊 {t('messages.searched_all', count=len(df_med))}")
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
                    st.info(f"📊 {t('messages.matched_cluster', cluster_id=cluster_id)}")

        # Display results
        if recommendations.empty:
            st.warning(t("messages.no_recommendations"))
            return

        st.markdown("---")
        st.subheader(f"✨ {t('sections.top_results', count=len(recommendations))}")

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

                col_a, col_b, col_c, col_d = st.columns([1.5, 1.5, 2, 1])

                with col_a:
                    st.write(f"**{t('labels.type')}:** {summary['type']}")

                with col_b:
                    # Display photo
                    img_url = strain_full.get('img_url', '')
                    if img_url and pd.notna(img_url):
                        st.markdown(f"**{t('labels.photo')}:**")
                        st.image(img_url, width=100)
                    else:
                        st.write(f"**{t('labels.photo')}:** N/A")

                with col_c:
                    st.write(f"**{t('labels.thc_level')}:** {summary['thc_level']}")

                with col_d:
                    st.write(f"**{t('labels.distance')}:** {summary['match_score']}")
                    st.caption(t("labels.lower_is_better"))

                # Positive effects
                if summary['positive_effects']:
                    st.markdown(f"**✅ {t('labels.positive_effects')}**")
                    st.write(", ".join(summary['positive_effects']))
                else:
                    st.write(f"**✅ {t('labels.positive_effects')}** {t('labels.none_above_threshold')}")

                # Negative effects
                if summary['negative_effects']:
                    st.markdown(f"**⚠️ {t('labels.negative_effects')}**")
                    st.write(", ".join(summary['negative_effects']))
                else:
                    st.write(f"**⚠️ {t('labels.negative_effects')}** {t('labels.none_above_threshold')}")

                st.markdown("---")

    # Footer
    st.markdown("")
    st.markdown("")
    with st.expander(f"ℹ️ {t('about.title')}"):
        st.markdown(f"""
        {t('about.intro')}

        - {t('about.kmeans')}
        - {t('about.knn')}

        {t('about.data_info')}

        {t('about.distance_title')}
        {t('about.distance_lower')}
        {t('about.distance_not_percent')}
        {t('about.distance_top')}
        """)


if __name__ == "__main__":
    main()
