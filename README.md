# API de Crédito Consignado

Sistema de análise e predição de reincidência de crédito consignado usando Machine Learning.

## 📋 Descrição

Este projeto implementa uma solução completa para análise de crédito consignado, incluindo:
- Análise exploratória de dados
- Modelo de Machine Learning para predição de reincidência
- API REST para integração com sistemas externos

## 🗂️ Estrutura do Projeto

```
consignado-analytics/
│
├── data/                  # Dados do projeto (ignorados no Git)
│   ├── raw/               # Dados brutos (CSV original)
│   └── processed/         # Dados processados
│
├── models/                # Modelos treinados (.pkl - ignorados no Git)
│   ├── modelo_reincidencia_credito.pkl
│   └── encoders.pkl
│
├── notebooks/             # Notebooks Jupyter
│   └── Análise_de_reincidência_de_Crédito.ipynb
│
├── src/                   # Código fonte
│   ├── __init__.py
│   ├── preprocessing.py   # Funções de pré-processamento
│   ├── train.py           # Script de treinamento
│   └── api.py             # API FastAPI
│
├── .gitignore
├── requirements.txt       # Dependências do projeto
└── README.md
```

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone https://github.com/victorhprada/api-credito-consignado.git
cd api-credito-consignado
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env e adicione a URL do modelo no Google Drive
```

## 🔧 Dependências

- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- joblib
- pydantic
- gdown (para baixar o modelo)
- python-dotenv (para variáveis de ambiente)

## 📦 Configuração do Modelo

O modelo é grande demais para o GitHub (107 MB), então ele é baixado automaticamente do Google Drive.

### Como fazer upload do modelo no Google Drive:

1. Acesse https://drive.google.com
2. Faça upload do arquivo `modelo_reincidencia_credito.pkl`
3. Clique com botão direito no arquivo → "Compartilhar"
4. Em "Acesso geral", selecione "Qualquer pessoa com o link"
5. Copie o link compartilhado (formato: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
6. Use esse link na variável de ambiente `MODELO_URL`

### Configurar no Render:

1. Acesse seu projeto no Render
2. Vá em "Environment" → "Environment Variables"
3. Adicione a variável:
   - **Key**: `MODELO_URL`
   - **Value**: Link do Google Drive (formato completo)

## 📊 Uso

### Executar a API localmente

```bash
PYTHONPATH=consignado-analytics uvicorn consignado-analytics.src.api:app --reload
```

A API estará disponível em `http://localhost:8000`

### Documentação da API

Acesse `http://localhost:8000/docs` para ver a documentação interativa (Swagger UI).

### Deploy no Render

**Comando de Start:**
```bash
PYTHONPATH=consignado-analytics uvicorn consignado-analytics.src.api:app --host 0.0.0.0 --port $PORT
```

**Variáveis de Ambiente necessárias:**
- `MODELO_URL`: URL do modelo no Google Drive

## 📝 Licença

Este projeto é de uso pessoal/educacional.

## 👤 Autor

Victor Prada
