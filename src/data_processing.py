import os
import pandas as pd
import numpy as np
import re
from glob import glob

def load_and_merge_data():
    df_dengue = dengue_data()
    climate = climate_data()
    df_socio = socio_data()

    df_merged = merge_tables(df_dengue, climate, df_socio)
    df_processed = preprocess_data(df_merged)

    return df_processed

def preprocess_data(df_processed):
    df_processed['CASES_NOTIFIC'] = df_processed['CASES_NOTIFIC'].fillna(0)
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
    df_merged = pd.merge(df_climate, df_dengue, left_on=['Data Medicao', 'Cod IBGE'], right_on=['DATA', 'ID_MUNICIP'], how='left')
    df_merged = pd.merge(df_merged, df_socio, left_on='Cod IBGE', right_on='Código IBGE', how='left')

    df_merged = df_merged.drop(['DATA', 'ID_MUNICIP', 'Código IBGE', 'Cod IBGE'], axis=1)
    return df_merged

def socio_data():
    df_socio = pd.read_csv('data/raw/ips_brasil_municipios.csv')
    df_socio['Código IBGE'] = df_socio['Código IBGE'] // 10
    df_socio['Código IBGE'] = df_socio['Código IBGE'].astype(str)

    return df_socio

def dengue_data():
    # Busca todos os arquivos de dengue
    arquivos_csv = glob(os.path.join('data/raw/dengue', '*.csv'))

    dengue_dfs = []
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo, encoding='utf-8', low_memory=False, usecols=['DT_NOTIFIC', 'ID_MUNICIP'])
        df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], format='%Y%m%d', errors='coerce')
        df['ID_MUNICIP'] = df['ID_MUNICIP'].astype(str)
        
        df_dengue = df.dropna(subset=['DT_NOTIFIC']).groupby(['ID_MUNICIP', 'DT_NOTIFIC']).size().reset_index(name='CASES_NOTIFIC')
        df_dengue.rename(columns={'DT_NOTIFIC': 'DATA'}, inplace=True)
        dengue_dfs.append(df_dengue)
    
    dengue = pd.concat(dengue_dfs, ignore_index=True)
    dengue.to_csv('data/processed/dengue.csv', sep=';', index=False)
    return dengue

def climate_data():
    arquivos_csv = glob(os.path.join('data/raw/climate', '*.csv'))

    dataframes = []
    for arquivo in arquivos_csv:
    # Extrai o nome do arquivo (ex: "317010.csv" → "317010")
        nome_arquivo = os.path.basename(arquivo)
        sections = re.split(r'[.-]', nome_arquivo)
        cod_ibge = sections[0]

        df = pd.read_csv(arquivo, sep=';', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')
        df['Cod IBGE'] = cod_ibge

        columns_to_convert = [col for col in df.columns if col not in ['Data Medicao', 'Cod IBGE']]
        for col in columns_to_convert:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
            df[f'{col}_media_movel_7d'] = df[col].rolling(window=7, min_periods=1).mean().shift(1)
            df[f'{col}_media_movel_10d'] = df[col].rolling(window=10, min_periods=1).mean().shift(1)

        df = df.drop(columns=columns_to_convert)
        dataframes.append(df)

    climate = pd.concat(dataframes, ignore_index=True)
    climate = climate.bfill().ffill()
    climate.to_csv('data/processed/climate.csv', sep=';', index=False)

    return climate

def prepare_single_prediction_features(ibge_code, recent_dengue_cases, df_climate_recent):
    df_climate_recent['Cod IBGE'] = ibge_code
    df_climate_recent['DENG_CASES'] = recent_dengue_cases

    df_socio = socio_data()
    df_merged = pd.merge(df_climate_recent, df_socio, left_on='Cod IBGE', right_on='Código IBGE', how='left')

    df_processed = preprocess_data(df_merged)
    return df_processed

def process_data(filepath):
    df = pd.read_csv(filepath, sep=';', parse_dates=['Data_Medicao'], date_format='%Y-%m-%d')
    # Cria o alvo para 1 dia no futuro
    df = create_future_targets(df, 'CASES_NOTIFIC', horizon=1)
    df = create_lags(df, 'CASES_NOTIFIC', lags=[2, 4, 7])

    df['mes'] = df['Data_Medicao'].dt.month
    df['dia_da_semana'] = df['Data_Medicao'].dt.weekday
    df['sin_mes'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['cos_mes'] = np.cos(2 * np.pi * df['mes'] / 12)

    df = df.dropna()
    target_cols = ['CASES_NOTIFIC_+1']
    y = df[target_cols]
    X = df.drop(columns=['CASES_NOTIFIC', 'Data_Medicao'] + target_cols)

    return X, y

def create_future_targets(df, column_name, horizon):
    for i in range(1, horizon + 1):
        df[f'{column_name}_+{i}'] = df[column_name].shift(-i)
    return df

def create_lags(df, column_name, lags):
    for lag in lags:
        df[f'{column_name}_lag_{lag}'] = df[column_name].shift(lag)
    return df