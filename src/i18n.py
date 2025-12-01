"""
Sistema de internacionalização (i18n) para o Cannabis Strain Recommender.
Suporta múltiplos idiomas com carregamento dinâmico e cache.
"""

import json
import logging
import os
from typing import Dict, Any, List
import streamlit as st

logger = logging.getLogger(__name__)

# Diretório raiz do projeto
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALES_DIR = os.path.join(PROJECT_ROOT, "locales")


def load_translations(locale: str) -> Dict[str, Any]:
    """
    Carrega o arquivo de tradução para um idioma específico.

    Args:
        locale: Código do idioma (ex: "en", "pt_BR")

    Returns:
        Dicionário com as traduções

    Raises:
        FileNotFoundError: Se o arquivo de tradução não existir
        json.JSONDecodeError: Se o arquivo JSON for inválido
    """
    locale_file = os.path.join(LOCALES_DIR, f"{locale}.json")

    if not os.path.exists(locale_file):
        raise FileNotFoundError(f"Translation file not found: {locale_file}")

    with open(locale_file, "r", encoding="utf-8") as f:
        translations = json.load(f)

    logger.info(f"Loaded translations for locale: {locale}")
    return translations


def get_nested_value(data: Dict[str, Any], key_path: str) -> Any:
    """
    Obtém um valor de um dicionário usando notação de ponto.

    Args:
        data: Dicionário com os dados
        key_path: Caminho da chave usando pontos (ex: "app.title")

    Returns:
        Valor encontrado ou None se não existir
    """
    keys = key_path.split(".")
    value = data

    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None

    return value


def init_i18n(default_locale: str = "en") -> None:
    """
    Inicializa o sistema de i18n.

    Carrega as traduções e configura o idioma padrão no session_state.

    Args:
        default_locale: Idioma padrão a ser usado
    """
    # Inicializa o estado da sessão
    if "i18n_initialized" not in st.session_state:
        st.session_state.i18n_initialized = True
        st.session_state.current_locale = default_locale
        st.session_state.translations = {}

        # Carrega traduções para o idioma padrão
        try:
            st.session_state.translations[default_locale] = load_translations(default_locale)
            logger.info(f"i18n initialized with default locale: {default_locale}")
        except Exception as e:
            logger.error(f"Failed to load default translations: {e}")
            st.session_state.translations[default_locale] = {}


def get_current_locale() -> str:
    """
    Obtém o idioma atual da sessão.

    Returns:
        Código do idioma atual
    """
    return st.session_state.get("current_locale", "en")


def set_locale(locale: str) -> None:
    """
    Define o idioma atual da sessão.

    Carrega as traduções se ainda não estiverem em cache.

    Args:
        locale: Código do idioma a ser definido
    """
    if locale not in st.session_state.translations:
        try:
            st.session_state.translations[locale] = load_translations(locale)
        except Exception as e:
            logger.error(f"Failed to load translations for locale {locale}: {e}")
            return

    st.session_state.current_locale = locale
    logger.info(f"Locale changed to: {locale}")


def get_available_locales() -> List[str]:
    """
    Obtém lista de idiomas disponíveis.

    Returns:
        Lista de códigos de idiomas disponíveis
    """
    locales = []

    if not os.path.exists(LOCALES_DIR):
        logger.warning(f"Locales directory not found: {LOCALES_DIR}")
        return ["en"]

    for filename in os.listdir(LOCALES_DIR):
        if filename.endswith(".json"):
            locale = filename.replace(".json", "")
            locales.append(locale)

    return sorted(locales) if locales else ["en"]


def t(key: str, **kwargs) -> str:
    """
    Obtém uma string traduzida para o idioma atual.

    Suporta interpolação de variáveis usando placeholders {nome}.
    Implementa fallback de três níveis:
    1. Idioma atual
    2. Inglês (se disponível)
    3. Retorna a própria chave

    Args:
        key: Chave da tradução em notação de ponto (ex: "app.title")
        **kwargs: Variáveis para interpolação na string

    Returns:
        String traduzida com variáveis interpoladas

    Examples:
        >>> t("app.title")
        "Cannabis Strain Recommender"

        >>> t("sections.top_results", count=10)
        "Top 10 Recommendations"
    """
    current_locale = get_current_locale()

    # Nível 1: Tenta idioma atual
    if current_locale in st.session_state.translations:
        value = get_nested_value(st.session_state.translations[current_locale], key)
        if value is not None:
            # Interpola variáveis se fornecidas
            if kwargs:
                try:
                    return value.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing variable in translation key '{key}': {e}")
                    return value
            return value

    # Nível 2: Fallback para inglês
    if current_locale != "en" and "en" in st.session_state.translations:
        value = get_nested_value(st.session_state.translations["en"], key)
        if value is not None:
            logger.warning(f"Translation key '{key}' not found for locale '{current_locale}', using English")
            if kwargs:
                try:
                    return value.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing variable in translation key '{key}': {e}")
                    return value
            return value

    # Nível 3: Retorna a própria chave
    logger.warning(f"Translation key '{key}' not found in any locale, returning key itself")
    return key


# Alias para facilitar o uso
_ = t


if __name__ == "__main__":
    # Testa o sistema de i18n
    logging.basicConfig(level=logging.INFO)

    # Simula inicialização do Streamlit
    class MockSessionState:
        def __init__(self):
            self.data = {}

        def get(self, key, default=None):
            return self.data.get(key, default)

        def __setitem__(self, key, value):
            self.data[key] = value

        def __getitem__(self, key):
            return self.data[key]

        def __contains__(self, key):
            return key in self.data

    # Substitui st.session_state por mock para teste
    st.session_state = MockSessionState()

    print("\n=== Testing i18n System ===\n")

    # Testa inicialização
    print("1. Testing initialization...")
    init_i18n(default_locale="en")
    print(f"   Current locale: {get_current_locale()}")
    print(f"   Available locales: {get_available_locales()}")

    # Testa tradução básica
    print("\n2. Testing basic translation...")
    print(f"   t('app.title'): {t('app.title')}")

    # Testa tradução com variáveis
    print("\n3. Testing translation with variables...")
    print(f"   t('sections.top_results', count=10): {t('sections.top_results', count=10)}")

    # Testa fallback
    print("\n4. Testing fallback for missing key...")
    print(f"   t('non.existent.key'): {t('non.existent.key')}")

    # Testa mudança de idioma
    print("\n5. Testing locale change...")
    set_locale("pt_BR")
    print(f"   Current locale: {get_current_locale()}")
    print(f"   t('app.title'): {t('app.title')}")

    print("\n=== Tests completed ===\n")
