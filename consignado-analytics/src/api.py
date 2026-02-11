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
    # 1. Validação Básica
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Arquivo deve ser um CSV")
    
    try:
        # Leitura do arquivo
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        # --- ETAPA 1: MAPEAMENTO DE COLUNAS (NORMALIZAÇÃO) ---
        column_map = {}
        targets_found = set()

        for col in df.columns:
            col_lower = col.lower()
            target = None

            # Lógica de Identificação
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
            elif ('estado' in col_lower or 'uf' in col_lower) and 'civil' not in col_lower:
                target = 'estado'
            elif 'genero' in col_lower or 'sexo' in col_lower:
                target = 'genero'
            elif 'escolaridade' in col_lower:
                target = 'escolaridade'

            if target and target not in targets_found:
                column_map[col] = target
                targets_found.add(target)

        # Aplica renomeação e remove duplicatas
        df_processed = df.rename(columns=column_map).copy()
        
        # Garante apenas uma coluna de cada tipo (se houver duplicidade no CSV)
        df_final = pd.DataFrame()
        for target in targets_found:
            if target in df_processed.columns:
                if isinstance(df_processed[target], pd.DataFrame):
                    df_final[target] = df_processed[target].iloc[:, 0]
                else:
                    df_final[target] = df_processed[target]
        df_processed = df_final

        # --- ETAPA 2: LIMPEZA E CÁLCULOS (CLEANING) ---
        
        # Preenche colunas obrigatórias com padrão se não existirem
        defaults = {
            'dependentes': 0, 'anos_empresa': 0, 
            'estado': 'SP', 'genero': 'M', 
            'escolaridade': '2o Grau Completo', 'est_civil': 'Solteiro(a)',
            'salario': 0, 'idade_raw': None
        }
        for col, val in defaults.items():
            if col not in df_processed.columns:
                df_processed[col] = val

        # A. Cálculos Numéricos
        # Idade
        if 'idade_raw' in df_processed.columns:
            df_processed['idade'] = df_processed['idade_raw'].apply(calculate_age)
        elif 'idade' not in df_processed.columns:
            df_processed['idade'] = 0 

        # Salário
        df_processed['salario'] = df_processed['salario'].apply(clean_currency)
        
        # Dependentes
        df_processed['dependentes'] = df_processed['dependentes'].apply(clean_dependents)
            
        # Anos de Empresa
        df_processed['anos_empresa'] = df_processed['anos_empresa'].apply(calculate_years_worked)

        # --- ETAPA 3: ENCODING (TEXTO -> NÚMERO) ---
        # ATENÇÃO: Convertemos aqui para os números exatos que o modelo espera (0, 1, 2...)
        
        # Estado Civil
        mapa_civil = {
            'Casado(a)': 1, 'União Estável': 1,
            'Solteiro(a)': 0, 'Outros': 0,
            'Divorciado(a)': 2, 'Separado(a)': 2,
            'Viúvo(a)': 3
        }
        df_processed['est_civil_cod'] = df_processed['est_civil'].map(mapa_civil).fillna(0).astype(int)

        # Gênero
        mapa_sexo = {'M': 1, 'F': 0} 
        df_processed['genero_cod'] = df_processed['genero'].str.upper().map(mapa_sexo).fillna(0).astype(int)

        # Escolaridade
        mapa_escolaridade = {
            'Superior Completo': 3,
            '2º Grau Completo': 2, '2o Grau Completo': 2, 'Ensino Médio Completo': 2,
            '2º Grau Incompleto': 1, '2o Grau Incompleto': 1,
            'Fundamental': 0, '1o Grau Completo': 0, 'Alfabetizado': 0
        }
        df_processed['escolaridade_cod'] = df_processed['escolaridade'].map(mapa_escolaridade).fillna(0).astype(int)

        # Estado (UF) -> Mantendo mapeamento numérico (0-26)
        # ufs = sorted(['AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO'])
        # mapa_uf = {uf: i for i, uf in enumerate(ufs)}
        # df_processed['estado_cod'] = df_processed['estado'].str.upper().map(mapa_uf).fillna(0).astype(int)
        df_processed['estado_cod'] = 0

        

        print(f"Colunas disponíveis no DF Processado: {df_processed.columns.tolist()}")

        # --- ETAPA 4: PREPARAÇÃO FINAL PARA O MODELO ---
        
        # 1. Selecionar APENAS as colunas numéricas (CODIFICADAS)
        # IMPORTANTE: Aqui usamos os nomes '_cod' que acabamos de criar
        features_para_modelo = [
            'salario',          # Numérico
            'idade',            # Numérico
            'dependentes',      # Numérico
            'anos_empresa',     # Numérico
            'est_civil_cod',    # Codificado (0, 1, 2...)
            'genero_cod',       # Codificado (0, 1)
            'escolaridade_cod', # Codificado (0, 1, 2, 3)
            'estado_cod'        # Codificado (0-26)
        ]
        
        # Cria o DataFrame X limpo
        X = df_processed[features_para_modelo].copy()

        # 2. Renomear para os nomes EXATOS que o .pkl exige
        # De: "Nosso Nome Interno" -> Para: "Nome do Treino"
        rename_map = {
            'salario': 'Salario Base',
            'idade': 'Idade',
            'dependentes': 'Total De Dependentes',
            'anos_empresa': 'Anos_de_Empresa',
            'est_civil_cod': 'Estado Civil',        # Note: Liga o CODIFICADO ao nome final
            'genero_cod': 'Genero',                 # Note: Liga o CODIFICADO ao nome final
            'escolaridade_cod': 'Nivel De Escolaridade', # Note: Liga o CODIFICADO ao nome final
            'estado_cod': 'Estado'                  # Note: Liga o CODIFICADO ao nome final
        }
        
        X_final = X.rename(columns=rename_map)

        # 3. Ordenar colunas (Ordem obrigatória do Scikit-Learn)
        colunas_ordenadas = [
            'Salario Base', 
            'Idade', 
            'Total De Dependentes', 
            'Anos_de_Empresa', 
            'Estado Civil', 
            'Genero', 
            'Nivel De Escolaridade', 
            'Estado'
        ]
        
        # Reorganiza
        X_final = X_final[colunas_ordenadas]

        print(f"Colunas finais enviadas para predição: {X_final.columns.tolist()}")
        print(f"Exemplo de dados: {X_final.head(1).values}")

        # --- ÁREA DE DEBUG (RAIO-X) ---
        print("\n" + "="*30)
        print("RAIO-X DOS DADOS (O que o modelo está vendo):")
        
        # 1. Ver se tem valores zerados demais
        print("\nEstatísticas (Veja se a média do Salario/Idade faz sentido):")
        print(X_final.describe().to_string()) 

        # 2. Ver a primeira linha real
        print("\nPrimeira linha exata enviada:")
        print(X_final.iloc[0].to_dict())
        
        # 3. Ver se existem NaNs (Valores vazios que viraram 0 ou erro)
        print("\nTem valores nulos/NaN?")
        print(X_final.isna().sum().to_string())
        print("="*30 + "\n")
        # ------------------------------

        # --- ETAPA 5: PREDIÇÃO ---
        print("Indo para predição...")
        predictions = modelo.predict_proba(X_final)[:, 1]
        print("Predição concluída com sucesso!")

        # --- ETAPA 6: RETORNO ---
        # Monta o CSV de resposta usando o DF original processado (para o usuário ler os textos, não os códigos)
        df_retorno = df_processed.copy()
        
        # Limpa as colunas auxiliares para não poluir o CSV do usuário
        cols_sujeira = ['est_civil_cod', 'genero_cod', 'escolaridade_cod', 'estado_cod', 'idade_raw']
        df_retorno = df_retorno.drop(columns=[c for c in cols_sujeira if c in df_retorno.columns], errors='ignore')

        df_retorno['Probabilidade_Retencao'] = (predictions * 100).round(2)
        df_retorno['Classificacao'] = np.where(predictions > 0.3, 'Perfil Tomador', 'Propensão à Quitação')

        stream = io.StringIO()
        df_retorno.to_csv(stream, index=False)
        
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=analise_processada.csv"
        return response

    except Exception as e:
        erro_completo = traceback.format_exc()
        print(f"ERRO FATAL: {erro_completo}")
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)