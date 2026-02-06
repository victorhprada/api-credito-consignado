# src/api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
from src.preprocessing import limpar_dados_para_modelo # Importando do seu módulo

app = FastAPI()

# Carregar modelo (caminho relativo à pasta raiz quando rodar)
modelo = joblib.load('models/modelo_reincidencia_credito.pkl')

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