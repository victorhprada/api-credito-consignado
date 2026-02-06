# src/api.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import gdown
import os
from pathlib import Path
from src.preprocessing import limpar_dados_para_modelo # Importando do seu módulo

app = FastAPI()

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (ideal para APIs públicas)
    # Para produção mais restrita, use:
    # allow_origins=["https://retentionml.lovable.app"],
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todos os headers
)

# Função para baixar o modelo do Google Drive
def baixar_modelo():
    modelo_path = Path('consignado-analytics/models/modelo_reincidencia_credito.pkl')
    
    # Se o modelo já existe, não precisa baixar
    if modelo_path.exists():
        print("✓ Modelo encontrado localmente")
        return str(modelo_path)
    
    # Criar diretório se não existir
    modelo_path.parent.mkdir(parents=True, exist_ok=True)
    
    # URL do Google Drive (usar variável de ambiente)
    gdrive_url = os.getenv('MODELO_URL')
    
    if not gdrive_url:
        raise ValueError("MODELO_URL não configurada! Configure a variável de ambiente no Render.")
    
    print("⏬ Baixando modelo do Google Drive...")
    gdown.download(gdrive_url, str(modelo_path), quiet=False, fuzzy=True)
    print("✓ Modelo baixado com sucesso!")
    
    return str(modelo_path)

# Carregar modelo (baixa do Google Drive se necessário)
modelo_path = baixar_modelo()
modelo = joblib.load(modelo_path)

class DadosInput(BaseModel):
    salario: float
    idade: int
    dependentes: int
    anos_empresa: int
    estado: str
    genero: str
    escolaridade: str
    est_civil: str

@app.post("/predict")
def predict(dados: DadosInput):
    # 1. Transforma JSON em DataFrame
    df_raw = pd.DataFrame([{
        'Salario Base': dados.salario,
        'Idade': dados.idade,
        'Total De Dependentes': dados.dependentes,
        'Anos_de_Empresa': dados.anos_empresa,
        'Estado': dados.estado,
        'Genero': dados.genero,
        'Nivel De Escolaridade': dados.escolaridade,
        'Estado Civil': dados.est_civil
    }])

    # 2. Usa sua função de limpeza centralizada
    df_clean = limpar_dados_para_modelo(df_raw)

    # 3. Predição
    prob = modelo.predict_proba(df_clean)[0][1]
    
    return {
        "probabilidade": float(prob),
        "classificacao": "Retenção" if prob > 0.5 else "Churn"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)