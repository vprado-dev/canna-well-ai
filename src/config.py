"""
Configuração e constantes para o Sistema de Recomendação de Variedades de Cannabis.
Extraído do notebook Jupyter (canna_well_ai.ipynb).
"""

from typing import List

# Condições médicas (39 no total - todas mantidas com 0% de filtragem)
MEDICAL_COLS: List[str] = [
    "anxiety", "stress", "pain", "depression", "insomnia", "ptsd",
    "fatigue", "lack_of_appetite", "headaches", "bipolar_disorder",
    "cancer", "cramps", "gastrointestinal_disorder", "inflammation",
    "muscle_spasms", "eye_pressure", "migraines", "asthma", "anorexia",
    "arthritis", "add/adhd", "muscular_dystrophy", "hypertension",
    "glaucoma", "pms", "seizures", "spasticity", "spinal_cord_injury",
    "fibromyalgia", "crohn's_disease", "phantom_limb_pain", "epilepsy",
    "multiple_sclerosis", "parkinson's", "tourette's_syndrome",
    "alzheimer's", "hiv/aids", "tinnitus", "nausea",
]

# Efeitos positivos (13 no total - o que os usuários QUEREM sentir)
POS_EFFECT_COLS: List[str] = [
    "relaxed", "happy", "euphoric", "uplifted", "sleepy", "hungry",
    "creative", "energetic", "giggly", "focused", "aroused", "talkative",
    "tingly",
]

# Efeitos negativos (6 no total - o que os usuários QUEREM EVITAR)
NEG_EFFECT_COLS: List[str] = [
    "dry_mouth", "dry_eyes", "dizzy", "anxious", "paranoid", "headache",
]

# Todas as colunas de porcentagem (para pré-processamento de dados)
ALL_PERCENT_COLS: List[str] = MEDICAL_COLS + POS_EFFECT_COLS + NEG_EFFECT_COLS

# Colunas de características para KNN (58 no total: 39 médicas + 13 positivas + 6 negativas)
KNN_FEATURE_COLS: List[str] = MEDICAL_COLS + POS_EFFECT_COLS + NEG_EFFECT_COLS

# Parâmetros do Modelo
KMEANS_K: int = 6  # Número de clusters
KMEANS_RANDOM_STATE: int = 42
KMEANS_N_INIT: int = 10

KNN_N_NEIGHBORS: int = 10  # Número máximo de vizinhos para KNN
KNN_METRIC: str = "euclidean"

# Limiar de filtragem de dados
DISEASE_FILTER_THRESHOLD: float = 0.95  # Remove doenças com >95% de zeros (nenhuma no nosso caso)

# Caminhos de Arquivos
DATA_PATH: str = "data/leafly_strain_data.csv"
MODELS_DIR: str = "models/"

# Colunas de exibição para a interface
DISPLAY_COLS: List[str] = ["name", "type", "thc_level"]

# Limiar de exibição de efeitos (mostrar apenas efeitos acima desta porcentagem)
EFFECTS_DISPLAY_THRESHOLD: float = 10.0
