"""
Módulo de treinamento e persistência de modelos.
Gerencia o salvamento e carregamento de modelos treinados para inicialização rápida do app.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from typing import Dict
import logging

from src.config import (
    MODELS_DIR,
    MEDICAL_COLS,
    KNN_FEATURE_COLS,
    KNN_N_NEIGHBORS,
    KNN_METRIC
)
from src.clustering import train_kmeans

logger = logging.getLogger(__name__)


def ensure_models_dir(models_dir: str = MODELS_DIR) -> None:
    """
    Garante que o diretório de modelos existe.

    Args:
        models_dir: Caminho para o diretório de modelos
    """
    os.makedirs(models_dir, exist_ok=True)
    logger.info(f"Models directory ensured: {models_dir}")


def train_and_save_models(
    df: pd.DataFrame,
    models_dir: str = MODELS_DIR
) -> None:
    """
    Treina todos os modelos e salva em disco.

    Treina e salva 4 arquivos:
    - kmeans_model.pkl: Modelo de clustering KMeans (k=6)
    - scaler_kmeans.pkl: StandardScaler para características médicas (39)
    - knn_model.pkl: Modelo NearestNeighbors em todas as variedades
    - scaler_knn.pkl: StandardScaler para características KNN (58)

    Args:
        df: DataFrame pré-processado com todos os dados de variedades
        models_dir: Diretório para salvar arquivos de modelos
    """
    ensure_models_dir(models_dir)

    logger.info("Starting model training...")

    # 1. Treina K-Means em características médicas
    logger.info("Training K-Means clustering...")
    X_medical = df[MEDICAL_COLS].values
    kmeans, scaler_kmeans = train_kmeans(X_medical)

    # Adiciona atribuições de cluster ao DataFrame (necessário para KNN)
    X_medical_scaled = scaler_kmeans.transform(X_medical)
    df["cluster"] = kmeans.predict(X_medical_scaled)

    # Salva modelos K-Means
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_kmeans_path = os.path.join(models_dir, "scaler_kmeans.pkl")

    joblib.dump(kmeans, kmeans_path)
    joblib.dump(scaler_kmeans, scaler_kmeans_path)

    logger.info(f"Saved K-Means model to {kmeans_path}")
    logger.info(f"Saved K-Means scaler to {scaler_kmeans_path}")

    # 2. Treina KNN em características combinadas
    logger.info("Training KNN model...")
    X_knn = df[KNN_FEATURE_COLS].values

    # Ajusta scaler
    scaler_knn = StandardScaler()
    X_knn_scaled = scaler_knn.fit_transform(X_knn)

    # Treina KNN
    knn_model = NearestNeighbors(
        n_neighbors=KNN_N_NEIGHBORS,
        metric=KNN_METRIC
    )
    knn_model.fit(X_knn_scaled)

    # Salva modelos KNN
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    scaler_knn_path = os.path.join(models_dir, "scaler_knn.pkl")

    joblib.dump(knn_model, knn_path)
    joblib.dump(scaler_knn, scaler_knn_path)

    logger.info(f"Saved KNN model to {knn_path}")
    logger.info(f"Saved KNN scaler to {scaler_knn_path}")

    logger.info("Model training complete!")


def load_models(models_dir: str = MODELS_DIR) -> Dict:
    """
    Carrega todos os modelos treinados do disco.

    Args:
        models_dir: Diretório contendo arquivos de modelos

    Returns:
        Dict contendo:
            - kmeans: Modelo KMeans treinado
            - scaler_kmeans: StandardScaler para clustering
            - knn_model: Modelo NearestNeighbors treinado
            - scaler_knn: StandardScaler para KNN

    Raises:
        FileNotFoundError: Se algum arquivo de modelo estiver faltando
    """
    logger.info(f"Loading models from {models_dir}...")

    # Define caminhos dos arquivos
    kmeans_path = os.path.join(models_dir, "kmeans_model.pkl")
    scaler_kmeans_path = os.path.join(models_dir, "scaler_kmeans.pkl")
    knn_path = os.path.join(models_dir, "knn_model.pkl")
    scaler_knn_path = os.path.join(models_dir, "scaler_knn.pkl")

    # Verifica se todos os arquivos existem
    required_files = [kmeans_path, scaler_kmeans_path, knn_path, scaler_knn_path]
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Model file not found: {file_path}\n"
                f"Please run train_models.py first to generate model files."
            )

    # Carrega modelos
    models = {
        "kmeans": joblib.load(kmeans_path),
        "scaler_kmeans": joblib.load(scaler_kmeans_path),
        "knn_model": joblib.load(knn_path),
        "scaler_knn": joblib.load(scaler_knn_path),
    }

    logger.info("All models loaded successfully")
    return models


def get_model_info(models_dir: str = MODELS_DIR) -> Dict:
    """
    Obtém informações sobre modelos salvos.

    Args:
        models_dir: Diretório contendo arquivos de modelos

    Returns:
        Dict com informações dos arquivos de modelos (tamanhos, existência)
    """
    files = {
        "kmeans_model.pkl": os.path.join(models_dir, "kmeans_model.pkl"),
        "scaler_kmeans.pkl": os.path.join(models_dir, "scaler_kmeans.pkl"),
        "knn_model.pkl": os.path.join(models_dir, "knn_model.pkl"),
        "scaler_knn.pkl": os.path.join(models_dir, "scaler_knn.pkl"),
    }

    info = {}
    total_size = 0

    for name, path in files.items():
        if os.path.exists(path):
            size = os.path.getsize(path)
            info[name] = {
                "exists": True,
                "size_bytes": size,
                "size_mb": size / (1024 * 1024)
            }
            total_size += size
        else:
            info[name] = {"exists": False, "size_bytes": 0, "size_mb": 0}

    info["total_size_mb"] = total_size / (1024 * 1024)

    return info


if __name__ == "__main__":
    # Testa persistência de modelos
    logging.basicConfig(level=logging.INFO)

    from src.preprocess import preprocess_data

    # Carrega dados
    print("Carregando e pré-processando dados...")
    df = preprocess_data()

    # Treina e salva modelos
    print("\nTreinando e salvando modelos...")
    train_and_save_models(df)

    # Mostra informações dos modelos
    print("\nInformações dos Modelos:")
    info = get_model_info()
    for name, data in info.items():
        if name == "total_size_mb":
            print(f"\nTamanho total: {data:.2f} MB")
        elif data["exists"]:
            print(f"  {name}: {data['size_mb']:.2f} MB")
        else:
            print(f"  {name}: NÃO ENCONTRADO")

    # Testa carregamento
    print("\nTestando carregamento de modelos...")
    models = load_models()
    print(f"Carregados {len(models)} modelos com sucesso")
    print(f"  - Clusters K-Means: {models['kmeans'].n_clusters}")
    print(f"  - Vizinhos KNN: {models['knn_model'].n_neighbors}")
