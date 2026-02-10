# src/api.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import uvicorn
import gdown
import os
from pathlib import Path
from src.preprocessing import limpar_dados_para_modelo # Importando do seu módulo
from datetime import datetime
from io import StringIO
from fastapi.responses import StreamingResponse
import numpy as np
import io
import traceback

app = FastAPI()

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

def clean_currency(x):
    """Limpa moedas brasileiras (R$ 1.500,00 -> 1500.00)"""
    if isinstance(x, (int, float)):
        return x
    if pd.isna(x) or x == '':
        return 0
    clean_str = str(x).replace('R$', '').replace(' ', '').replace('.', '').replace(',', '.')
    try:
        val = float(clean_str)
        # Regra de sanidade: Se > 50k, provavelmente perdeu a virgula dos centavos
        if val > 50000:
            return val / 100
        return val
    except:
        return 0

def calculate_age(dob):
    """Calcula idade baseada em data ISO ou BR"""
    if pd.isna(dob):
        return 0
    try:
        # Tenta converter para datetime (aceita ISO e BR automaticamente)
        birth_date = pd.to_datetime(dob, errors='coerce') # dayfirst ajuda com formatos BR

        if pd.isna(birth_date):
            return 0

        today = datetime.now()

        if birth_date.tzinfo:
            birth_date = birth_date.tz_localize(None)

        age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
        return max(0, age) # Evita idade negativa
    except:
        return 0

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    # 1. Ler o arquivo CSV enviado
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um CSV")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # 2. Mapeamento Inteligente de Colunas com trava anti-duplicidade
        column_map = {}
        targets_found = set ()


        for col in df.columns:
            col_lower = col.lower()
            target = None
            if 'nascimento' in col_lower or 'birth' in col_lower:
                target = 'idade_raw'
            elif ('salario' in col_lower or 'base' in col_lower or 'income' in col_lower) and 'liquido' not in col_lower:
                target = 'salario'
            elif 'dependentes' in col_lower:
                target = 'dependentes'
            elif 'empresa' in col_lower or 'casa' in col_lower or 'admissao' in col_lower:
                target = 'anos_empresa'
            elif 'estado civil' in col_lower or 'civil' in col_lower:
                target = 'est_civil'
            # Só aceita "estado" se NÃO tiver "civil" no nome
            elif ('estado' in col_lower or 'uf' in col_lower) and 'civil' not in col_lower:
                target = 'estado'
            elif 'genero' in col_lower or 'sexo' in col_lower:
                target = 'genero'
            elif 'escolaridade' in col_lower:
                target = 'escolaridade'

            if target and target not in targets_found:
                column_map[col] = target
                targets_found.add(target)

        # Renomeia as colunas encontradas
        df_processed = df.rename(columns=column_map).copy()

        # Limpeza Extra: Garante que só temos UMA coluna de cada, sem duplicatas
        df_final = pd.DataFrame()
        # Copia apenas as colunas que mapeamos
        for target in targets_found:
            if target in df_processed.columns:
                # Se ainda assim houver duplicata, pega só a primeira
                if isinstance(df_processed[target], pd.DataFrame):
                    df_final[target] = df_processed[target].iloc[:, 0]
                else:
                    df_final[target] = df_processed[target]
        
        df_processed = df_final

        # 3. Tratamento de Dados (Cleaning)
        # Calcula Idade se necessário
        if 'idade_raw' in df_processed.columns:
            df_processed['idade'] = df_processed['idade_raw'].apply(calculate_age)
        elif 'idade' not in df_processed.columns:
            df_processed['idade'] = 0 # Fallback

        # Limpa Salário
        if 'salario' in df_processed.columns:
            df_processed['salario'] = df_processed['salario'].apply(clean_currency)
        else:
            df_processed['salario'] = 0

        # Garante que as outras colunas existam (com valores padrao)
        defaults = {
            'dependentes': 0, 'anos_empresa': 0, 
            'estado': 'SP', 'genero': 'M', 
            'escolaridade': 'Indefinido', 'est_civil': 'Solteiro(a)'
        }
        for col, val in defaults.items():
            if col not in df_processed.columns:
                df_processed[col] = val

        print(f"Colunas processadas: {df_processed.columns.tolist()}")

        # 4. Prepara para o Modelo (A ordem das colunas importa!)
        # IMPORTANTE: Usa a mesma ordem que usou no treino do modelo para evitar erros de ordem
        features = ['salario', 'idade', 'dependentes', 'anos_empresa', 'estado', 'genero', 'escolaridade', 'est_civil']
        X = df_processed[features]

        rename_map = {
            'salario': 'Salario',           # ATENÇÃO: Se no treino era "Salario Base", mude aqui!
            'idade': 'Idade',
            'dependentes': 'Dependentes',
            'anos_empresa': 'Anos_de_Empresa', # O log mostrou que este tem underscores
            'estado': 'Estado',
            'genero': 'Genero',
            'escolaridade': 'Escolaridade',
            'est_civil': 'Estado Civil'
        }

        X_final = X.rename(columns=rename_map)
        cols_model_order = ['salario', 'idade', 'estado', 'anos_empresa', 'dependentes','escolaridade', 'genero',  'est_civil']
        X_final = X_final[cols_model_order]

        print(f"Colunas enviadas para o modelo: {X_final.columns.tolist()}")

        print("Indo para predição")

        # 5. Predição em Lote (Vetorizada)
        predictions = modelo.predict_proba(X_final)[:, 1]
        print("Predição concluída com sucesso!")

        # 6. Adiciona resultados ao DataFrame original
        df_retorno = df_processed.copy()
        df_retorno['Probabilidade_Retencao'] = (predictions * 100).round(2)
        df_retorno['Classificacao'] = np.where(predictions > 0.5, 'Perfil Tomador', 'Propensão à Quitação')

        # 7. Retorna o CSV modificado
        stream = io.StringIO()
        df_retorno.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=analise_processada.csv"
        return response

    except Exception as e:
        erro_completo = traceback.format_exc()
        print(f"Erro completo: {erro_completo}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar CSV: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)