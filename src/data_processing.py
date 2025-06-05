import os
import pandas as pd
import numpy as np
import re
from glob import glob

def preprocess_data():
    # Lendo todos os arquivos
    df_dengue = dengue_data()
    climate = climate_data()
    df_socio = socio_data()

    # Mergeando as três tabelas
    df_processed = merge_tables(df_dengue, climate, df_socio)

    # Ajustando nome das colunas
    df_processed['DENG_CASES'] = df_processed['DENG_CASES'].fillna(0)
    df_processed.columns = (
        df_processed.columns
        .astype(str)
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )

    # Convertendo dados das colunas
    for col in df_processed.columns:
        if df_processed[col].dtype == 'object':
            df_processed[col] = (
                df_processed[col]
                .astype(str)
                .str.replace(',', '.', regex=False)
                .str.extract(r'([-+]?\d*\.?\d+)', expand=False)
            )
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed = df_processed.replace([np.inf, -np.inf], np.nan)
    df_processed = df_processed.fillna(0)

    df_processed.to_csv('data/processed/table_processed.csv', sep=';', index=False)

    return df_processed

def merge_tables(df_dengue, df_climate, df_socio):
    df_processed = pd.merge(df_climate, df_dengue, left_on=['Data Medicao', 'Cod IBGE'], right_on=['DT_NOTIFIC', 'ID_MUNICIP'], how='left')
    df_processed = pd.merge(df_processed, df_socio, left_on='Cod IBGE', right_on='Código IBGE', how='left')

    df_processed = df_processed.drop(['DT_NOTIFIC', 'ID_MUNICIP', 'Código IBGE', 'Município', 'UF'], axis=1)
    return df_processed

def socio_data():
    #Preprocessing city table
    df_socio = pd.read_csv('data/raw/ips_brasil_municipios.csv')
    df_socio['Código IBGE'] = df_socio['Código IBGE'] // 10
    df_socio['Código IBGE'] = df_socio['Código IBGE'].astype(str)

    return df_socio

def dengue_data():
    # Busca todos os arquivos de dengue
    arquivos_csv = glob(os.path.join('data/raw/dengue', '*.csv'))

    dataframes = []
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo, parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False)
        df['ID_MUNICIP'] = df['ID_MUNICIP'].astype(str)
        df = df[['DT_NOTIFIC', 'ID_MUNICIP']].groupby(['ID_MUNICIP', 'DT_NOTIFIC']).size().reset_index(name='DENG_CASES')
        
        dataframes.append(df)
    
    dengue = pd.concat(dataframes, ignore_index=True)
    dengue.to_csv('data/processed/dengue.csv', sep=';', index=False)
    return dengue

def climate_data():
    # Busca todos os arquivos de climate
    arquivos_csv = glob(os.path.join('data/raw/climate', '*.csv'))

    dataframes = []
    for arquivo in arquivos_csv:
    # Extrai o nome do arquivo (ex: "317010.csv" → "317010")
        nome_arquivo = os.path.basename(arquivo)
        sections = re.split(r'[.-]', nome_arquivo)
        cod_ibge = sections[0]

        df = pd.read_csv(arquivo, sep=';', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')
        df['Cod IBGE'] = cod_ibge

        dataframes.append(df)

    climate = pd.concat(dataframes, ignore_index=True)

    climate.to_csv('data/processed/climate.csv', sep=';', index=False)
    return climate