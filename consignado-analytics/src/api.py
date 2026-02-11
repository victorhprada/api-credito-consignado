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
    allow_origins=["https://retentionml.lovable.app"],
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

def clean_dependents(x):
    """
    Converte 'Cônjuge; Pai/Mãe' -> 2
    Converte 'Filho(a)' -> 1
    Converte 3 -> 3
    """
    if pd.isna(x) or x == '':
        return 0
    
    # Se já for número, retorna int
    if isinstance(x, (int, float)):
        return int(x)
        
    s = str(x).strip()
    
    # Se for "0", "1", "10"...
    if s.isdigit():
        return int(s)
        
    # Lógica de Contagem de Lista (separada por ; ou ,)
    # Ex: "Pai; Mãe" tem 1 ponto e vírgula, logo são 2 pessoas
    delimiters = [';', ',']
    for char in delimiters:
        if char in s:
            return s.count(char) + 1
            
    # Se for texto sem separador (ex: "Filho"), assumimos 1
    if len(s) > 0:
        return 1
        
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

def calculate_years_worked(val):
    """
    Calcula anos de casa baseado na data de admissão.
    Entrada: '2020-01-01' -> Saída: 4.1
    Entrada: 5 -> Saída: 5.0
    """
    if pd.isna(val) or val == '':
        return 0
    
    # Se já vier como número (ex: planilha calculada), retorna o número
    if isinstance(val, (int, float)):
        return float(val)
        
    # Se for string numérica (ex: "5.5")
    s = str(val).strip()
    try:
        if s.replace('.', '', 1).isdigit():
            return float(s)
    except:
        pass

    try:
        # Tenta converter string de data para objeto datetime
        admission_date = pd.to_datetime(val, errors='coerce')
        
        if pd.isna(admission_date):
            return 0
            
        # Remove fuso horário se tiver, para evitar erro de comparação
        if admission_date.tzinfo:
            admission_date = admission_date.tz_localize(None)
            
        today = datetime.now()
        
        # Cálculo de dias / 365.25
        delta = today - admission_date
        years = delta.days / 365.25
        
        return max(0.0, round(years, 2)) # Retorna arredondado, sem negativo
    except:
        return 0

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um CSV")
    
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # --- 1. MAPEAMENTO DE COLUNAS ---
        column_map = {}
        targets_found = set()
        
        # Mapeamento flexível (aceita vários nomes)
        for col in df.columns:
            c = col.lower()
            target = None
            if 'nascimento' in c or 'birth' in c: target = 'idade_raw'
            elif ('salario' in c or 'base' in c or 'income' in c) and 'liquido' not in c: target = 'salario'
            elif 'dependentes' in c: target = 'dependentes'
            elif 'empresa' in c or 'admissao' in c: target = 'anos_empresa'
            elif 'estado civil' in c or 'civil' in c: target = 'est_civil'
            elif ('estado' in c or 'uf' in c) and 'civil' not in c: target = 'estado'
            elif 'genero' in c or 'sexo' in c: target = 'genero'
            elif 'escolaridade' in c: target = 'escolaridade'

            if target and target not in targets_found:
                column_map[col] = target
                targets_found.add(target)

        df_processed = df.rename(columns=column_map).copy()
        
        # Garante colunas únicas
        df_final = pd.DataFrame()
        for t in targets_found:
            if t in df_processed.columns:
                if isinstance(df_processed[t], pd.DataFrame):
                    df_final[t] = df_processed[t].iloc[:, 0]
                else:
                    df_final[t] = df_processed[t]
        df_processed = df_final

        # --- 2. VALORES PADRÃO & LIMPEZA ---
        defaults = {
            'dependentes': 0, 'anos_empresa': 0, 'estado': 'SP', 
            'genero': 'M', 'escolaridade': '2o Grau Completo', 
            'est_civil': 'Solteiro(a)', 'salario': 0, 'idade_raw': None
        }
        for col, val in defaults.items():
            if col not in df_processed.columns: df_processed[col] = val

        # Cálculos
        if 'idade_raw' in df_processed.columns:
            df_processed['idade'] = df_processed['idade_raw'].apply(calculate_age)
        elif 'idade' not in df_processed.columns:
            df_processed['idade'] = 0

        df_processed['salario'] = df_processed['salario'].apply(clean_currency)
        df_processed['dependentes'] = df_processed['dependentes'].apply(clean_dependents)
        df_processed['anos_empresa'] = df_processed['anos_empresa'].apply(calculate_years_worked)

        # --- 3. ENCODING (ORDEM ALFABÉTICA - CRÍTICO!) ---
        # O LabelEncoder usa ordem alfabética. Se trocarmos isso, o modelo erra.
        
        # Estado Civil (Ordem Alfabética Padrão)
        # 0: Casado, 1: Divorciado, 2: Outros, 3: Separado, 4: Solteiro, 5: União, 6: Viúvo
        mapa_civil = {
            'Casado(a)': 0, 'União Estável': 5,
            'Solteiro(a)': 4, 'Outros': 2,
            'Divorciado(a)': 1, 'Separado(a)': 3,
            'Viúvo(a)': 6
        }
        # Normaliza e mapeia (Fallback para 4=Solteiro se não achar)
        df_processed['est_civil_cod'] = df_processed['est_civil'].map(mapa_civil).fillna(4).astype(int)

        # Gênero (F vem antes de M)
        mapa_sexo = {'F': 0, 'M': 1} 
        df_processed['genero_cod'] = df_processed['genero'].str.upper().map(mapa_sexo).fillna(1).astype(int)

        # Escolaridade (Alfabética)
        # Cuidado: "1º" vem antes de "2º", "Superior" vem por último.
        # Ajuste conforme os dados usados no treino. Assumindo lista padrão:
        mapa_escolaridade = {
            '1º Grau Completo': 0, '1o Grau Completo': 0,
            '1º Grau Incompleto': 1, '1o Grau Incompleto': 1,
            '2º Grau Completo': 2, '2o Grau Completo': 2, 'Ensino Médio Completo': 2,
            '2º Grau Incompleto': 3, '2o Grau Incompleto': 3, 'Ensino Médio Incompleto': 3,
            'Alfabetizado': 4,
            'Analfabeto': 5,
            'Superior Completo': 6,
            'Superior Incompleto': 7
        }
        df_processed['escolaridade_cod'] = df_processed['escolaridade'].map(mapa_escolaridade).fillna(2).astype(int)

        # Estado UF (Alfabética 0-26)
        # O LabelEncoder numerou AC=0, AL=1... SP=25.
        ufs = sorted(['AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO'])
        mapa_uf = {uf: i for i, uf in enumerate(ufs)}
        df_processed['estado_cod'] = df_processed['estado'].str.upper().map(mapa_uf).fillna(25).astype(int)

        # --- 4. PREPARAÇÃO FINAL ---
        features_para_modelo = [
            'salario', 'idade', 'dependentes', 'anos_empresa',
            'est_civil_cod', 'genero_cod', 'escolaridade_cod', 'estado_cod'
        ]
        X = df_processed[features_para_modelo].copy()

        rename_map = {
            'salario': 'Salario Base', 'idade': 'Idade', 
            'dependentes': 'Total De Dependentes', 'anos_empresa': 'Anos_de_Empresa',
            'est_civil_cod': 'Estado Civil', 'genero_cod': 'Genero', 
            'escolaridade_cod': 'Nivel De Escolaridade', 'estado_cod': 'Estado'
        }
        X_final = X.rename(columns=rename_map)
        
        # Ordem Obrigatória
        colunas_ordenadas = [
            'Salario Base', 'Idade', 'Total De Dependentes', 'Anos_de_Empresa', 
            'Estado Civil', 'Genero', 'Nivel De Escolaridade', 'Estado'
        ]
        X_final = X_final[colunas_ordenadas]

        # --- 5. PREDIÇÃO & RETORNO ---
        print("Indo para predição...")
        probs = modelo.predict_proba(X_final)[:, 1]
        
        # RAIO-X LOG
        print(f"Probabilidade Média: {(probs.mean() * 100):.2f}%")
        print(f"Tomadores (>50%): {(probs > 0.5).sum()} de {len(probs)}")

        df_retorno = df_processed.copy()
        
        # Limpeza visual
        cols_sujeira = ['est_civil_cod', 'genero_cod', 'escolaridade_cod', 'estado_cod', 'idade_raw']
        df_retorno = df_retorno.drop(columns=[c for c in cols_sujeira if c in df_retorno.columns], errors='ignore')

        # [CORREÇÃO] Nome da coluna deve ser 'Probabilidade' para o Front reconhecer
        df_retorno['Probabilidade'] = (probs * 100).round(2)
        df_retorno['Classificacao'] = np.where(probs > 0.5, 'Perfil Tomador', 'Propensão à Quitação')

        stream = io.StringIO()
        df_retorno.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=analise_processada.csv"
        return response

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)