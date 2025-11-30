"""
Módulo de pré-processamento de dados para o Sistema de Recomendação de Variedades de Cannabis.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

from src.config import (
    DATA_PATH,
    ALL_PERCENT_COLS,
    MEDICAL_COLS
)

logger = logging.getLogger(__name__)


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Carrega o dataset bruto de variedades de cannabis a partir de CSV.

    Args:
        path: Caminho para o arquivo CSV

    Returns:
        pd.DataFrame: Dataset bruto com todas as colunas

    Raises:
        FileNotFoundError: Se o arquivo CSV não existir
    """
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} strains from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found: {path}")
        raise


def convert_percentages_to_float(
    df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Converte strings de porcentagem para valores float.

    Remove símbolos '%', converte para float e preenche NaN com 0.0.

    Args:
        df: DataFrame com colunas de porcentagem
        columns: Lista de nomes de colunas para converter

    Returns:
        pd.DataFrame: DataFrame com colunas convertidas
    """
    df_copy = df.copy()

    for col in columns:
        if col not in df_copy.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue

        df_copy[col] = (
            df_copy[col]
            .astype(str)
            .str.rstrip('%')
            .replace('', np.nan)
            .astype(float)
            .fillna(0.0)
        )

    logger.info(f"Converted {len(columns)} percentage columns to float")
    return df_copy


def convert_thc_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte o nível de THC de string de porcentagem para numérico.

    Args:
        df: DataFrame com coluna thc_level

    Returns:
        pd.DataFrame: DataFrame com coluna thc_level_num adicionada
    """
    df_copy = df.copy()

    df_copy["thc_level_num"] = (
        df_copy["thc_level"]
        .astype(str)
        .str.rstrip('%')
        .replace('', np.nan)
        .astype(float)
        .fillna(0.0)
    )

    logger.info("Converted THC level to numeric")
    return df_copy


def filter_medical_strains(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra variedades para manter apenas aquelas com pelo menos um valor médico > 0.
    - Isso reduz o dataset de ~4,764 para ~2,921 variedades.
    Args:
        df: DataFrame com colunas de condições médicas
    Returns:
        pd.DataFrame: DataFrame filtrado com índice resetado
    """
    X_raw = df[MEDICAL_COLS].values
    mask = X_raw.sum(axis=1) > 0
    df_filtered = df.loc[mask].reset_index(drop=True)

    logger.info(f"Filtered to {len(df_filtered)} strains with medical values (from {len(df)} total)")
    return df_filtered


def preprocess_data(data_path: str = DATA_PATH) -> pd.DataFrame:
    """
    Pipeline completo de pré-processamento para dados de variedades.
    Orquestra o processo completo de limpeza de dados:
    1. Carrega CSV bruto
    2. Converte colunas de porcentagem para floats
    3. Converte nível de THC para numérico
    4. Filtra para variedades com valores médicos
    Args:
        data_path: Caminho para o arquivo CSV
    Returns:
        pd.DataFrame: Dataset totalmente pré-processado pronto para modelagem
                     (~2,921 variedades com todas as porcentagens como floats)
    """
    logger.info("Starting data preprocessing pipeline")

    df = load_raw_data(data_path)

    df = convert_percentages_to_float(df, ALL_PERCENT_COLS)

    df = convert_thc_level(df)

    df = filter_medical_strains(df)

    logger.info(f"Preprocessing complete. Final shape: {df.shape}")
    return df


if __name__ == "__main__":
    # Testa o pipeline de pré-processamento
    logging.basicConfig(level=logging.INFO)
    df = preprocess_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nMedical columns sample:\n{df[MEDICAL_COLS].head()}")
