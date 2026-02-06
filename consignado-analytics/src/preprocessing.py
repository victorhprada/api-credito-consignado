# src/preprocessing.py
import pandas as pd
import numpy as np

def limpar_dados_para_modelo(df_input):
    """
    Recebe um DataFrame bruto (ou input da API) e devolve pronto para o modelo.
    """
    df = df_input.copy()
    
    # Mapeamentos (Copie exatamente o que você definiu no Colab)
    map_sexo = {'M': 1, 'F': 0}
    map_est_civil = {'Casado(a)': 1, 'Solteiro(a)': 0, 'Divorciado(a)': 2, 'Viúvo(a)': 3}
    map_escolaridade = {'Superior Completo': 3, '2º Grau Completo': 2, '2º Grau Incompleto': 1, 'Fundamental': 0}

    # Aplicação segura dos mapas
    # Nota: Em produção, idealmente usamos LabelEncoders salvos, mas hardcoded funciona para MVP
    if 'Genero' in df.columns:
        df['Genero'] = df['Genero'].map(map_sexo).fillna(0)
    
    if 'Estado Civil' in df.columns:
        df['Estado Civil'] = df['Estado Civil'].map(map_est_civil).fillna(0)

    if 'Nivel De Escolaridade' in df.columns:
        df['Nivel De Escolaridade'] = df['Nivel De Escolaridade'].map(map_escolaridade).fillna(0)

    # Tratamento de Estado (Simplificado para MVP - ideal é usar encoder treinado)
    if 'Estado' in df.columns:
        # Aqui você pode precisar de uma lógica mais robusta ou carregar o encoder
        # Exemplo simples: Transformar em categórico numérico arbitrário se não tiver o encoder
        df['Estado'] = 0 

    # Garantir a ordem das colunas EXATAMENTE como no treino
    cols_ordem = ['Salario Base', 'Idade', 'Total De Dependentes', 'Anos_de_Empresa', 
                  'Estado Civil', 'Genero', 'Nivel De Escolaridade', 'Estado']
    
    # Filtra apenas as colunas necessárias e retorna
    return df[cols_ordem]