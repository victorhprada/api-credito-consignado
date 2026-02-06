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

## 🔧 Dependências

- pandas
- numpy
- scikit-learn
- fastapi
- uvicorn
- joblib
- pydantic

## 📊 Uso

### Executar a API

```bash
uvicorn src.api:app --reload
```

A API estará disponível em `http://localhost:8000`

### Documentação da API

Acesse `http://localhost:8000/docs` para ver a documentação interativa (Swagger UI).

## 📝 Licença

Este projeto é de uso pessoal/educacional.

## 👤 Autor

Victor Prada
