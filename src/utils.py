"""
Funções utilitárias para formatação de exibição e logging.
"""

import pandas as pd
from typing import List
import logging

from src.config import EFFECTS_DISPLAY_THRESHOLD

logger = logging.getLogger(__name__)


def format_strain_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formata dados de variedades para exibição limpa na interface.

    Trata níveis de THC ausentes exibindo como "N/A".

    Args:
        df: DataFrame com dados de variedades

    Returns:
        pd.DataFrame: DataFrame formatado
    """
    df_display = df.copy()

    # Trata níveis de THC ausentes
    if "thc_level" in df_display.columns:
        df_display["thc_level"] = df_display["thc_level"].fillna("N/A")
        # Também trata strings vazias
        df_display["thc_level"] = df_display["thc_level"].replace("", "N/A")

    # Trata tipos ausentes
    if "type" in df_display.columns:
        df_display["type"] = df_display["type"].fillna("Unknown")
        df_display["type"] = df_display["type"].replace("", "Unknown")

    return df_display


def log_medical_scores(
    strain_row: pd.Series,
    medical_cols: List[str]
) -> None:
    """
    Registra pontuações de efetividade médica para uma variedade.

    Isto é apenas para depuração/análise - NÃO é exibido ao usuário final.

    Args:
        strain_row: Uma única linha do DataFrame de variedades
        medical_cols: Lista de nomes de colunas de condições médicas
    """
    strain_name = strain_row.get("name", "Unknown")

    # Extrai pontuações médicas
    medical_scores = {}
    for col in medical_cols:
        if col in strain_row.index:
            score = strain_row[col]
            if score > 0:
                medical_scores[col] = score

    if medical_scores:
        # Ordena por pontuação decrescente
        sorted_scores = sorted(medical_scores.items(), key=lambda x: x[1], reverse=True)
        top_5 = sorted_scores[:5]

        logger.info(f"Medical scores for '{strain_name}': "
                   f"{', '.join([f'{cond}={score:.1f}%' for cond, score in top_5])}")
    else:
        logger.info(f"No medical scores for '{strain_name}'")


def format_effects_list(
    strain_row: pd.Series,
    effect_cols: List[str],
    threshold: float = EFFECTS_DISPLAY_THRESHOLD
) -> List[str]:
    """
    Extrai efeitos acima do limiar e formata como lista com porcentagens.

    Args:
        strain_row: Uma única linha do DataFrame de variedades
        effect_cols: Lista de nomes de colunas de efeitos para verificar
        threshold: Porcentagem mínima para incluir (padrão: 10.0)

    Returns:
        Lista de strings formatadas como ["Relaxed (66%)", "Happy (54%)"]
    """
    from src.i18n import t
    from src.config import POS_EFFECT_COLS, NEG_EFFECT_COLS, MEDICAL_COLS

    effects = []

    for col in effect_cols:
        if col in strain_row.index:
            value = strain_row[col]
            if value >= threshold:
                # Traduz o nome do efeito com base no tipo
                if col in POS_EFFECT_COLS:
                    translated_name = t(f"effects.positive.{col}")
                elif col in NEG_EFFECT_COLS:
                    translated_name = t(f"effects.negative.{col}")
                elif col in MEDICAL_COLS:
                    translated_name = t(f"medical_conditions.{col}")
                else:
                    # Fallback: formata nome da coluna como antes
                    translated_name = col.replace("_", " ").title()
                    translated_name = translated_name.replace("Add/Adhd", "ADD/ADHD")
                    translated_name = translated_name.replace("Hiv/Aids", "HIV/AIDS")

                effects.append(f"{translated_name} ({value:.0f}%)")

    # Ordena por porcentagem (decrescente)
    # Usa rsplit para pegar o último '(' que contém a porcentagem
    effects.sort(key=lambda x: float(x.rsplit("(", 1)[1].split("%")[0]), reverse=True)

    return effects


def format_match_score(distance: float) -> str:
    """
    Formata distância KNN como uma pontuação de correspondência amigável ao usuário.

    Menor distância = melhor correspondência.

    Args:
        distance: Distância euclidiana do KNN

    Returns:
        String formatada
    """
    return f"{distance:.2f}"


def get_strain_summary(
    strain_row: pd.Series,
    pos_effect_cols: List[str],
    neg_effect_cols: List[str],
    threshold: float = EFFECTS_DISPLAY_THRESHOLD
) -> dict:
    """
    Obtém um resumo completo de uma variedade para exibição.

    Args:
        strain_row: Uma única linha do DataFrame de variedades
        pos_effect_cols: Lista de colunas de efeitos positivos
        neg_effect_cols: Lista de colunas de efeitos negativos
        threshold: Porcentagem mínima para incluir efeitos

    Returns:
        Dict com informações formatadas da variedade
    """
    summary = {
        "name": strain_row.get("name", "Unknown"),
        "type": strain_row.get("type", "Unknown"),
        "thc_level": strain_row.get("thc_level", "N/A"),
        "match_score": format_match_score(strain_row.get("knn_distance", 0.0)),
        "positive_effects": format_effects_list(strain_row, pos_effect_cols, threshold),
        "negative_effects": format_effects_list(strain_row, neg_effect_cols, threshold),
    }

    # Trata valores ausentes
    if summary["type"] == "" or pd.isna(summary["type"]):
        summary["type"] = "Unknown"

    if summary["thc_level"] == "" or pd.isna(summary["thc_level"]):
        summary["thc_level"] = "N/A"

    return summary


if __name__ == "__main__":
    # Testa funções utilitárias
    logging.basicConfig(level=logging.INFO)

    from src.preprocess import preprocess_data
    from src.config import MEDICAL_COLS, POS_EFFECT_COLS, NEG_EFFECT_COLS

    # Carrega dados
    df = preprocess_data()

    # Testa com primeira variedade
    strain = df.iloc[0]
    print(f"\nStrain: {strain['name']}")

    # Testa format_effects_list
    pos_effects = format_effects_list(strain, POS_EFFECT_COLS)
    print(f"Positive effects: {pos_effects}")

    neg_effects = format_effects_list(strain, NEG_EFFECT_COLS)
    print(f"Negative effects: {neg_effects}")

    # Testa log_medical_scores
    log_medical_scores(strain, MEDICAL_COLS)

    # Testa get_strain_summary
    strain["knn_distance"] = 5.74  # Distância simulada
    summary = get_strain_summary(strain, POS_EFFECT_COLS, NEG_EFFECT_COLS)
    print(f"\nStrain Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
