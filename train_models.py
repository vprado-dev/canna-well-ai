"""
Script único para treinar e salvar modelos.

Execute este script uma vez para gerar os arquivos de modelos necessários para o app Streamlit.

Uso:
    python train_models.py
"""

import logging
import sys

from src.preprocess import preprocess_data
from src.models import train_and_save_models, get_model_info

# Configura logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Pipeline principal de treinamento."""
    logger.info("=" * 60)
    logger.info("Recomendador de Variedades de Cannabis - Treinamento de Modelos")
    logger.info("=" * 60)

    try:
        # Etapa 1: Carrega e pré-processa dados
        logger.info("\nEtapa 1: Carregando e pré-processando dados...")
        df = preprocess_data()
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns[:10])}...")  # Mostra primeiras 10 colunas

        # Etapa 2: Treina e salva modelos
        logger.info("\nEtapa 2: Treinando e salvando modelos...")
        train_and_save_models(df)

        # Etapa 3: Verifica modelos salvos
        logger.info("\nEtapa 3: Verificando modelos salvos...")
        model_info = get_model_info()

        print("\n" + "=" * 60)
        print("Treinamento de Modelos Concluído!")
        print("=" * 60)
        print("\nModelos Salvos:")

        for name, info in model_info.items():
            if name == "total_size_mb":
                print(f"\nTamanho Total: {info:.2f} MB")
            elif info["exists"]:
                print(f"  ✓ {name:<25} {info['size_mb']:.2f} MB")
            else:
                print(f"  ✗ {name:<25} NÃO ENCONTRADO")

        print("\nAgora você pode executar o app Streamlit:")
        print("  streamlit run app.py")
        print("=" * 60)

        return 0

    except FileNotFoundError as e:
        logger.error(f"\nErro: {e}")
        logger.error("Certifique-se de que o arquivo de dados existe no caminho correto")
        return 1

    except Exception as e:
        logger.error(f"\nErro inesperado: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
