"""
Algoritmo de recomendação KNN para o Sistema de Recomendação de Variedades de Cannabis.
Extraído das células 15-17 do notebook.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import List, Tuple
import logging

from src.config import (
    KNN_FEATURE_COLS,
    MEDICAL_COLS,
    KNN_N_NEIGHBORS,
    KNN_METRIC
)
from src.clustering import assign_user_to_cluster

logger = logging.getLogger(__name__)


def build_user_vector(
    diseases: List[str] = None,
    desired_effects: List[str] = None,
    avoid_effects: List[str] = None,
) -> np.ndarray:
    """
    Constrói um vetor de preferências do usuário a partir de condições e efeitos selecionados.

    Extraído da célula 15 do notebook.

    Args:
        diseases: Lista de condições médicas para tratar (definido como 100.0)
        desired_effects: Lista de efeitos positivos desejados (definido como 100.0)
        avoid_effects: Lista de efeitos negativos a evitar (definido como 0.0)

    Returns:
        np.ndarray: Vetor do usuário de forma (1, 58) com valores 0.0 ou 100.0
    """
    diseases = diseases or []
    desired_effects = desired_effects or []
    avoid_effects = avoid_effects or []

    # Inicializa todas as características como 0.0
    user_pref = pd.Series(0.0, index=KNN_FEATURE_COLS)

    # Define doenças como 100.0
    for disease in diseases:
        if disease in user_pref.index:
            user_pref[disease] = 100.0
        else:
            logger.warning(f"Disease '{disease}' not in KNN_FEATURE_COLS")

    # Define efeitos desejados como 100.0
    for effect in desired_effects:
        if effect in user_pref.index:
            user_pref[effect] = 100.0
        else:
            logger.warning(f"Desired effect '{effect}' not in KNN_FEATURE_COLS")

    # Efeitos a evitar permanecem em 0.0 (queremos variedades com valores baixos para estes)
    for effect in avoid_effects:
        if effect not in user_pref.index:
            logger.warning(f"Avoid effect '{effect}' not in KNN_FEATURE_COLS")

    logger.info(f"Built user vector: {len(diseases)} diseases, "
                f"{len(desired_effects)} desired effects, "
                f"{len(avoid_effects)} avoid effects")

    return user_pref.values.reshape(1, -1)


def recommend_strains_global_knn(
    user_vector: np.ndarray,
    knn_model: NearestNeighbors,
    scaler_knn: StandardScaler,
    df: pd.DataFrame,
    n_neighbors: int = KNN_N_NEIGHBORS
) -> pd.DataFrame:
    """
    Recomenda variedades usando KNN global (busca em todas as variedades).

    Extraído da célula 16 do notebook.

    Args:
        user_vector: Vetor de preferências do usuário (1, 58)
        knn_model: Modelo NearestNeighbors treinado em todas as variedades
        scaler_knn: StandardScaler ajustado para características KNN
        df: DataFrame com todos os dados de variedades
        n_neighbors: Número de recomendações para retornar

    Returns:
        pd.DataFrame: Variedades recomendadas com colunas:
                     name, type, thc_level, knn_distance
    """
    # Escala vetor do usuário
    user_vec_scaled = scaler_knn.transform(user_vector)

    # Encontra vizinhos mais próximos
    distances, indices = knn_model.kneighbors(
        user_vec_scaled,
        n_neighbors=n_neighbors
    )

    # Constrói DataFrame de recomendações
    recs = df.iloc[indices[0]].copy()
    recs["knn_distance"] = distances[0]

    # Seleciona colunas de exibição
    display_cols = ["name", "type", "thc_level", "knn_distance"]
    if "cluster" in recs.columns:
        display_cols.append("cluster")

    # Filtra para colunas disponíveis
    display_cols = [col for col in display_cols if col in recs.columns]
    recs = recs[display_cols]

    # Ordena por distância (mais próximo primeiro)
    recs = recs.sort_values("knn_distance").reset_index(drop=True)

    logger.info(f"Global KNN: Found {len(recs)} recommendations")
    return recs


def recommend_strains_cluster_knn(
    user_vector: np.ndarray,
    diseases: List[str],
    kmeans: KMeans,
    scaler_kmeans: StandardScaler,
    scaler_knn: StandardScaler,
    df: pd.DataFrame,
    n_neighbors: int = KNN_N_NEIGHBORS
) -> pd.DataFrame:
    """
    Recomenda variedades usando KNN baseado em cluster.

    Processo em duas etapas:
    1. Atribui usuário a um cluster com base em condições médicas
    2. Busca vizinhos mais próximos dentro desse cluster

    Extraído da célula 17 do notebook.

    Args:
        user_vector: Vetor de preferências do usuário (1, 58) para busca KNN
        diseases: Lista de condições médicas do usuário para atribuição de cluster
        kmeans: Modelo KMeans treinado
        scaler_kmeans: StandardScaler para clustering (39 características)
        scaler_knn: StandardScaler para KNN (58 características)
        df: DataFrame com todos os dados de variedades e atribuições de cluster
        n_neighbors: Número de recomendações para retornar

    Returns:
        pd.DataFrame: Variedades recomendadas do cluster do usuário
    """
    # Etapa 1: Atribui usuário a um cluster com base apenas em doenças
    user_cluster = assign_user_to_cluster(diseases, kmeans, scaler_kmeans)
    logger.info(f"User assigned to cluster {user_cluster}")

    # Filtra DataFrame para o cluster do usuário
    df_cluster = df[df["cluster"] == user_cluster].copy()

    if df_cluster.empty:
        logger.warning(f"Cluster {user_cluster} is empty, falling back to global KNN")
        # Fallback: usa KNN global pré-treinado
        # Por enquanto, retorna DataFrame vazio
        return pd.DataFrame(columns=["name", "type", "thc_level", "cluster", "knn_distance"])

    # Etapa 2: Constrói KNN local nas variedades do cluster
    X_cluster = df_cluster[KNN_FEATURE_COLS].values
    X_cluster_scaled = scaler_knn.transform(X_cluster)

    # Ajusta n_neighbors se o cluster for menor
    n_neighbors_eff = min(n_neighbors, len(df_cluster))

    # Treina KNN local
    knn_local = NearestNeighbors(
        n_neighbors=n_neighbors_eff,
        metric=KNN_METRIC
    )
    knn_local.fit(X_cluster_scaled)

    # Escala vetor do usuário e busca
    user_vec_scaled = scaler_knn.transform(user_vector)
    distances, indices = knn_local.kneighbors(
        user_vec_scaled,
        n_neighbors=n_neighbors_eff
    )

    # Constrói DataFrame de recomendações
    recs = df_cluster.iloc[indices[0]].copy()
    recs["knn_distance"] = distances[0]

    # Seleciona colunas de exibição
    display_cols = ["name", "type", "thc_level", "cluster", "knn_distance"]
    display_cols = [col for col in display_cols if col in recs.columns]
    recs = recs[display_cols]

    # Ordena por distância
    recs = recs.sort_values("knn_distance").reset_index(drop=True)

    logger.info(f"Cluster KNN: Found {len(recs)} recommendations in cluster {user_cluster}")
    return recs


if __name__ == "__main__":
    # Testa módulo de recomendação
    logging.basicConfig(level=logging.INFO)

    from src.preprocess import preprocess_data
    from src.clustering import train_kmeans
    from sklearn.neighbors import NearestNeighbors

    # Carrega e pré-processa dados
    df = preprocess_data()

    # Treina clustering
    X_medical = df[MEDICAL_COLS].values
    kmeans, scaler_kmeans = train_kmeans(X_medical)
    X_medical_scaled = scaler_kmeans.transform(X_medical)
    df["cluster"] = kmeans.predict(X_medical_scaled)

    # Treina KNN global
    X_knn = df[KNN_FEATURE_COLS].values
    scaler_knn = StandardScaler()
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    knn_model = NearestNeighbors(n_neighbors=KNN_N_NEIGHBORS, metric=KNN_METRIC)
    knn_model.fit(X_knn_scaled)

    # Caso de teste da célula 19 do notebook
    print("\n=== Test Case: Anxiety + Depression ===")
    user_vec = build_user_vector(
        diseases=["anxiety", "depression"],
        desired_effects=["relaxed", "happy", "focused"],
        avoid_effects=["dry_mouth", "paranoid"]
    )

    print("\nGlobal KNN Recommendations:")
    recs_global = recommend_strains_global_knn(user_vec, knn_model, scaler_knn, df, n_neighbors=5)
    print(recs_global)

    print("\nCluster-based KNN Recommendations:")
    recs_cluster = recommend_strains_cluster_knn(
        user_vec, ["anxiety", "depression"],
        kmeans, scaler_kmeans, scaler_knn, df, n_neighbors=5
    )
    print(recs_cluster)
