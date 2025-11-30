"""
Módulo de clustering K-Means para o Sistema de Recomendação de Variedades de Cannabis.
Extraído das células 5-8 do notebook.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Tuple, List, Dict
import logging

from src.config import (
    MEDICAL_COLS,
    KMEANS_K,
    KMEANS_RANDOM_STATE,
    KMEANS_N_INIT
)

logger = logging.getLogger(__name__)


def train_kmeans(
    X: np.ndarray,
    k: int = KMEANS_K
) -> Tuple[KMeans, StandardScaler]:
    """
    Treina modelo de clustering K-Means em características médicas.

    Extraído das células 5-7 do notebook.

    Args:
        X: Matriz de características (n_samples, n_features) com valores de condições médicas
        k: Número de clusters (padrão: 6)

    Returns:
        Tupla de (modelo KMeans treinado, StandardScaler ajustado)
    """
    # Escala características
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Treina K-Means
    kmeans = KMeans(
        n_clusters=k,
        random_state=KMEANS_RANDOM_STATE,
        n_init=KMEANS_N_INIT
    )
    kmeans.fit(X_scaled)

    logger.info(f"K-Means trained with k={k} clusters")
    logger.info(f"Inertia: {kmeans.inertia_:.2f}")

    return kmeans, scaler


def assign_user_to_cluster(
    user_diseases: List[str],
    kmeans: KMeans,
    scaler: StandardScaler
) -> int:
    """
    Atribui um usuário a um cluster com base em suas condições médicas.

    Extraído da célula 17 do notebook.

    Args:
        user_diseases: Lista de nomes de doenças (ex: ["anxiety", "depression"])
        kmeans: Modelo KMeans treinado
        scaler: StandardScaler ajustado do treinamento

    Returns:
        int: ID do cluster (0 a k-1)
    """
    # Constrói vetor médico do usuário (todos zeros inicialmente)
    user_med = pd.Series(0.0, index=MEDICAL_COLS)

    # Define doenças selecionadas como 100.0
    for disease in user_diseases:
        if disease in user_med.index:
            user_med[disease] = 100.0
        else:
            logger.warning(f"Disease '{disease}' not in MEDICAL_COLS")

    # Converte para array e escala
    user_vec = user_med.values.reshape(1, -1)
    user_vec_scaled = scaler.transform(user_vec)

    # Prediz cluster
    cluster_id = int(kmeans.predict(user_vec_scaled)[0])

    logger.info(f"User assigned to cluster {cluster_id} based on {len(user_diseases)} diseases")
    return cluster_id


def get_cluster_profile(
    df: pd.DataFrame,
    cluster_id: int,
    top_n: int = 10
) -> Dict[str, float]:
    """
    Obtém o perfil de condições médicas para um cluster específico.

    Retorna as porcentagens médias de efetividade para condições médicas
    neste cluster, ordenadas por efetividade.

    Extraído da célula 8 do notebook.

    Args:
        df: DataFrame com atribuições de cluster e colunas de condições médicas
        cluster_id: O ID do cluster para perfilar
        top_n: Número de condições principais para retornar (padrão: 10)

    Returns:
        Dict mapeando nomes de condições para porcentagens médias de efetividade
    """
    cluster_df = df[df["cluster"] == cluster_id]

    if cluster_df.empty:
        logger.warning(f"Cluster {cluster_id} is empty")
        return {}

    # Calcula média para cada condição médica
    means = cluster_df[MEDICAL_COLS].mean().sort_values(ascending=False)

    # Retorna top N como dict
    profile = means.head(top_n).to_dict()

    logger.info(f"Cluster {cluster_id} profile: {len(cluster_df)} strains, "
                f"top condition: {means.index[0]} ({means.iloc[0]:.1f}%)")

    return profile


def get_cluster_distribution(df: pd.DataFrame) -> pd.Series:
    """
    Obtém a distribuição de variedades entre clusters.

    Args:
        df: DataFrame com atribuições de cluster

    Returns:
        pd.Series: Contagem de variedades por cluster
    """
    if "cluster" not in df.columns:
        raise ValueError("DataFrame must have 'cluster' column")

    distribution = df["cluster"].value_counts().sort_index()
    logger.info(f"Cluster distribution:\n{distribution}")

    return distribution


if __name__ == "__main__":
    # Testa módulo de clustering
    logging.basicConfig(level=logging.INFO)

    from src.preprocess import preprocess_data

    # Carrega dados
    df = preprocess_data()

    # Extrai características médicas
    X_raw = df[MEDICAL_COLS].values

    # Treina K-Means
    kmeans, scaler = train_kmeans(X_raw, k=6)

    # Atribui clusters às variedades
    X_scaled = scaler.transform(X_raw)
    df["cluster"] = kmeans.predict(X_scaled)

    # Mostra distribuição
    print("\nDistribuição de Clusters:")
    print(get_cluster_distribution(df))

    # Mostra perfil para cada cluster
    print("\nPerfis de Clusters:")
    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\n=== Cluster {cluster_id} ===")
        profile = get_cluster_profile(df, cluster_id, top_n=5)
        for condition, pct in profile.items():
            print(f"  {condition}: {pct:.1f}%")
