# 📊 Credit Retention Intelligence (CRI)

![Project Status](https://img.shields.io/badge/status-concluído-success)
![Python Version](https://img.shields.io/badge/python-3.10+-blue)
![Stack](https://img.shields.io/badge/stack-FullStack_Data_Science-orange)
[![Keep Alive Render](https://github.com/victorhprada/api-credito-consignado/actions/workflows/keep_alive.yml/badge.svg?branch=main)](https://github.com/victorhprada/api-credito-consignado/actions/workflows/keep_alive.yml)

> **Uma plataforma inteligente para predição de Churn (saída de clientes) em Crédito Consignado, capaz de processar grandes volumes de dados para apoiar decisões estratégicas.**

---

## 🎯 O Problema de Negócio

Empresas de crédito lidam com milhares de contratos ativos. Identificar quais clientes estão propensos a sair (quitar o contrato ou fazer portabilidade) é crucial para a retenção.
Anteriormente, essa análise era feita de forma **manual em planilhas Excel**, o que era:
* **Lento:** Demorava horas para processar 1.000 clientes.
* **Limitado:** Impossível analisar a base inteira (40.000+ clientes) de uma vez.
* **Subjetivo:** Baseado na intuição, não em dados estatísticos.

### 🚀 A Solução
Desenvolvi uma aplicação Web completa que utiliza **Inteligência Artificial** para ler o histórico do cliente e calcular a probabilidade exata dele manter o contrato.

**Resultados Alcançados:**
* ✅ **Escalabilidade:** Processamento de **45.000+ linhas** em poucos minutos.
* ✅ **Precisão:** Modelo de Machine Learning treinado com dados históricos reais.
* ✅ **Eficiência:** Redução drástica no tempo operacional da equipe de análise.

---

## 🛠️ Deep Dive Técnico (Para Tech Leads e Devs)

Este projeto não é apenas um modelo de ML, é uma aplicação **Full Stack de Ciência de Dados** projetada para contornar limitações reais de infraestrutura.

### 🏗️ Arquitetura e Stack
* **Frontend:** React (Vite) + TailwindCSS (Interface moderna e responsiva).
* **Backend:** Python com **FastAPI** (Alta performance e assincronismo).
* **Machine Learning:** Scikit-Learn (**Random Forest Classifier**), Pandas e Numpy.
* **Deploy:** Render (Cloud).

### 🔥 O Grande Desafio Técnico: "Big Data" no Free Tier
Um dos maiores desafios foi processar arquivos CSV gigantes (45k+ linhas) em um ambiente de nuvem com recursos limitados (512MB RAM e Timeouts curtos).

**A Solução de Engenharia:**
Implementei uma estratégia de **Client-Side Chunking (Fatiamento no Frontend)**:
1.  O Frontend lê o arquivo CSV localmente.
2.  Quebra os dados em "lotes" (chunks) de 1.000 linhas.
3.  Envia requisições sequenciais para a API Python.
4.  O Backend processa, prevê e retorna o lote.
5.  O Frontend remonta o arquivo final para o usuário.

> *Isso permitiu processar volumes ilimitados de dados sem estourar a memória do servidor e sem sofrer timeouts de conexão (Erro 504), garantindo uma experiência fluida.*

### ⚡ DevOps: Mantendo a API "Acordada"
Outro desafio do plano gratuito do Render é o **"Cold Start"**: o servidor desliga após 15 minutos de inatividade, causando lentidão na primeira requisição.

**A Solução de Automação:**
Implementei um workflow de CI/CD no **GitHub Actions** que atua como um *Heartbeat*:
1.  Um **Cron Job** é executado automaticamente a cada 14 minutos.
2.  Ele envia um "ping" leve para a rota de saúde (`/`) da API.
3.  Isso impede que o container hiberne, garantindo alta disponibilidade e resposta rápida a qualquer momento.

> *Arquivo de configuração: `.github/workflows/keep_alive.yml`*


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

---

## 🧠 O Modelo de Machine Learning

O coração do sistema é um algoritmo **Random Forest** que analisa padrões comportamentais.

**Pipeline de Dados (ETL):**
1.  **Limpeza:** Tratamento automático de moedas (`R$ 1.200,00` -> `1200.0`), datas e valores nulos.
2.  **Feature Engineering:** Cálculo automático de "Idade" e "Tempo de Casa" baseados nas datas.
3.  **Encoding:** Transformação inteligente de variáveis categóricas (Estado Civil, Escolaridade) respeitando a semântica dos dados.

---

## 📸 Screenshots

*(Espaço reservado para colocar os prints que você me mandou: A tela de upload, a barra de progresso funcionando e a tela de resultado final)*

---

## 🚀 Como Rodar o Projeto Localmente

### Pré-requisitos
* Python 3.10+
* Node.js 18+

### Passo 1: Backend (API)
```bash
# Clone o repositório
git clone [https://github.com/seu-usuario/consignado-analytics.git](https://github.com/seu-usuario/consignado-analytics.git)
cd consignado-analytics/src

# Crie e ative o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Rode a API
uvicorn api:app --reload
```

### Passo 2: Frontend (Interface)
```bash
# Em outro terminal, vá para a pasta do front
cd frontend

# Instale as dependências
npm install

# Rode o servidor de desenvolvimento
npm run dev
```

## 📝 Licença

Este projeto é de uso pessoal.

## 👤 Autor

**Victor Prada**

*Analista de Dados e Cientista de Dados*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/victorh-prada/)
