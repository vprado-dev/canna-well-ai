# 🌿 Recomendador de Variedades de Cannabis

Uma aplicação web alimentada por machine learning para recomendar variedades de cannabis com base em condições médicas e efeitos desejados.

## Estrutura do Projeto (Project Structure)

```
canna-well-ai/
├── .streamlit/
│   └── config.toml                 # Tema personalizado (verde claro)
├── data/
│   └── leafly_strain_data.csv    # Dataset bruto de variedades
├── models/                         # Modelos treinados (gerados por train_models.py)
│   ├── kmeans_model.pkl
│   ├── scaler_kmeans.pkl
│   ├── knn_model.pkl
│   └── scaler_knn.pkl
├── src/
│   ├── __init__.py
│   ├── config.py                   # Constantes e configuração
│   ├── preprocess.py               # Pipeline de pré-processamento de dados
│   ├── clustering.py               # Lógica de clustering K-Means
│   ├── recommender.py              # Motores de recomendação KNN
│   ├── models.py                   # Persistência de modelos
│   └── utils.py                    # Auxiliares de formatação de exibição
├── app.py                          # Aplicação Streamlit
├── train_models.py                 # Script de treinamento de modelos
├── requirements.txt                # Dependências Python
└── README.md                       # Este arquivo
```

## Stack Tecnológico / Technology Stack

- **Python 3.8+**
- **Streamlit**: Web app framework / Framework de aplicação web
- **scikit-learn**: Machine learning (K-Means, KNN)
- **pandas**: Data manipulation / Manipulação de dados
- **numpy**: Numerical operations / Operações numéricas
- **joblib**: Model persistence / Persistência de modelos

## Desempenho do Modelo / Model Performance

- **Silhouette Score (k=6)**: 0.5762 (good cluster quality / boa qualidade de cluster)
- **Calinski-Harabasz**: 94.32 (well-defined clusters / clusters bem definidos)
- **Model Files / Arquivos de Modelo**: ~1.3 MB total
- **App Startup / Inicialização do App**: < 1 second / < 1 segundo
- **Recommendation Time / Tempo de Recomendação**: < 1 second per request / < 1 segundo por solicitação

## Dataset

- **Source / Fonte**: Leafly strain data / Dados de variedades do Leafly
- **Dataset Link / Link do Dataset**: [Kaggle - Leafly Cannabis Strains Metadata](https://www.kaggle.com/datasets/gthrosa/leafly-cannabis-strains-metadata)
- **Total Strains / Total de Variedades**: 4,762 (filtered to 2,921 with medical data / filtradas para 2.921 com dados médicos)
- **Features / Características**: Medical conditions, positive effects, negative effects, THC levels, strain types / Condições médicas, efeitos positivos, efeitos negativos, níveis de THC, tipos de variedades

## Funcionalidades

- **Métodos de Recomendação Duplos**:
  - KNN Global: Busca em todas as 2.921 variedades
  - KNN Baseado em Cluster: Faz correspondência por perfil médico primeiro, depois busca dentro do cluster

- **Dados Abrangentes**: 39 condições médicas, 13 efeitos positivos, 6 efeitos negativos

- **Interface Interativa**: Construída com Streamlit para fácil uso

- **Alto Desempenho**: Modelos pré-treinados carregam instantaneamente


## Instalação

### Pré-requisitos

- Python 3.8 ou superior
- pip

### Configuração

1. Clone o repositório:
```bash
git clone <your-repo-url>
cd canna-well-ai
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Treine os modelos (configuração única):
```bash
python train_models.py
```

Isso criará arquivos de modelos treinados no diretório `models/` (~1.3 MB no total).

## Uso

### Execute o App Streamlit

```bash
streamlit run app.py
```

O app será aberto no seu navegador padrão em `http://localhost:8501`.

### Usando o App

1. **Escolha o Método de Recomendação**: Selecione entre KNN Global ou KNN Baseado em Cluster

2. **Selecione Condições Médicas**: Escolha uma ou mais condições que deseja tratar

3. **Escolha Efeitos Desejados**: Selecione efeitos positivos que você está procurando

4. **Selecione Efeitos a Evitar**: Escolha efeitos negativos que você deseja minimizar

5. **Defina Número de Recomendações**: Use o controle deslizante para escolher quantas variedades recomendar (5-20)

6. **Obtenha Recomendações**: Clique no botão para ver suas recomendações personalizadas

### Entendendo os Resultados

Cada variedade recomendada mostra:
- **Nome**: Nome da variedade
- **Tipo**: Indica, Sativa ou Híbrida
- **Nível de THC**: Porcentagem (ou N/A se desconhecido)
- **Pontuação de Correspondência**: Menor é melhor (distância euclidiana)
- **Efeitos Positivos**: Efeitos acima do limiar de 10% com porcentagens
- **Efeitos Negativos**: Efeitos colaterais acima do limiar de 10% com porcentagens

## Como Funciona

### 1. Pré-processamento de Dados
- Converte strings de porcentagem para floats
- Filtra para variedades com pelo menos um benefício médico
- Resulta em 2.921 variedades utilizáveis

### 2. Clustering K-Means (k=6)
- Agrupa variedades por perfis de efetividade médica
- Usa 39 características de condições médicas
- StandardScaler separado para clustering

### 3. Recomendações KNN

**KNN Global**:
- Constrói vetor do usuário a partir das seleções (58 características)
- Busca em todas as 2.921 variedades
- Retorna os N vizinhos mais próximos

**KNN Baseado em Cluster**:
- Atribui usuário a um cluster com base em condições médicas
- Filtra variedades para aquele cluster
- Busca dentro do cluster para melhores correspondências
- Melhor para necessidades médicas especializadas

### 4. Exibição de Resultados
- Mostra informações básicas (nome, tipo, THC)
- Lista efeitos positivos e negativos com porcentagens
- Registra pontuações médicas para análise (não mostrado ao usuário)

## Desenvolvimento

### Execute Testes

Teste módulos individuais:

```bash
# Testa pré-processamento
python src/preprocess.py

# Testa clustering
python src/clustering.py

# Testa recomendador
python src/recommender.py

# Testa utils
python src/utils.py

# Testa modelos
python src/models.py
```

### Retreinar Modelos

Se você atualizar os dados ou algoritmos:

```bash
python train_models.py
```

## Licença

[Adicione sua licença aqui]

## Agradecimentos

- Dataset do Leafly disponível no [Kaggle](https://www.kaggle.com/datasets/gthrosa/leafly-cannabis-strains-metadata)
- Construído com Streamlit
- Machine learning com scikit-learn

## Suporte

Para problemas ou perguntas, por favor abra uma issue no GitHub ou entre em contato [seu-contato].

---

**Nota**: Esta aplicação é apenas para fins educacionais e informativos. Sempre consulte profissionais de saúde para orientação médica.

---

# 🌿 Cannabis Strain Recommender

A machine learning-powered web application for recommending cannabis strains based on medical conditions and desired effects.

## Features

- **Dual Recommendation Methods**:
  - Global KNN: Searches across all 2,921 strains
  - Cluster-based KNN: Matches by medical profile first, then searches within cluster

- **Comprehensive Data**: 39 medical conditions, 13 positive effects, 6 negative effects

- **Interactive UI**: Built with Streamlit for easy use

- **Fast Performance**: Pre-trained models load instantly

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd canna-well-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the models (one-time setup):
```bash
python train_models.py
```

This will create trained model files in the `models/` directory (~1.3 MB total).

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Using the App

1. **Choose Recommendation Method**: Select between Global KNN or Cluster-based KNN

2. **Select Medical Conditions**: Pick one or more conditions you want to treat

3. **Choose Desired Effects**: Select positive effects you're looking for

4. **Select Effects to Avoid**: Pick negative effects you want to minimize

5. **Set Number of Recommendations**: Use the slider to choose how many strains to recommend (5-20)

6. **Get Recommendations**: Click the button to see your personalized recommendations

### Understanding Results

Each recommended strain shows:
- **Name**: Strain name
- **Type**: Indica, Sativa, or Hybrid
- **THC Level**: Percentage (or N/A if unknown)
- **Match Score**: Lower is better (Euclidean distance)
- **Positive Effects**: Effects above 10% threshold with percentages
- **Negative Effects**: Side effects above 10% threshold with percentages

## How It Works

### 1. Data Preprocessing
- Converts percentage strings to floats
- Filters to strains with at least one medical benefit
- Results in 2,921 usable strains

### 2. K-Means Clustering (k=6)
- Groups strains by medical effectiveness profiles
- Uses 39 medical condition features
- Separate StandardScaler for clustering

### 3. KNN Recommendations

**Global KNN**:
- Builds user vector from selections (58 features)
- Searches all 2,921 strains
- Returns top N nearest neighbors

**Cluster-based KNN**:
- Assigns user to cluster based on medical conditions
- Filters strains to that cluster
- Searches within cluster for best matches
- Better for specialized medical needs

### 4. Results Display
- Shows basic info (name, type, THC)
- Lists positive and negative effects with percentages
- Logs medical scores for analytics (not shown to user)

## Development

### Run Tests

Test individual modules:

```bash
# Test preprocessing
python src/preprocess.py

# Test clustering
python src/clustering.py

# Test recommender
python src/recommender.py

# Test utils
python src/utils.py

# Test models
python src/models.py
```

### Retrain Models

If you update the data or algorithms:

```bash
python train_models.py
```

## License

[Add your license here]

## Acknowledgments

- Dataset from Leafly available on [Kaggle](https://www.kaggle.com/datasets/gthrosa/leafly-cannabis-strains-metadata)
- Built with Streamlit
- Machine learning with scikit-learn

## Support

For issues or questions, please open an issue on GitHub or contact [your-contact-info].

---

**Note**: This application is for educational and informational purposes only. Always consult with healthcare professionals for medical advice.
