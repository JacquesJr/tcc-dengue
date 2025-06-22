import os
import pandas as pd
import numpy as np
import re
from glob import glob

def load_and_merge_data():
    # Lendo todos os arquivos
    df_dengue = dengue_data()
    climate = climate_data()
    df_socio = socio_data()

    # Mergeando as três tabelas
    df_merged = merge_tables(df_dengue, climate, df_socio)

    df_processed = preprocess_data(df_merged)
    return df_processed

def preprocess_data(df_processed):
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
    df_merged = pd.merge(df_climate, df_dengue, left_on=['Data Medicao', 'Cod IBGE'], right_on=['DT_SIN_PRI', 'ID_MUNICIP'], how='left')
    df_merged = pd.merge(df_merged, df_socio, left_on='Cod IBGE', right_on='Código IBGE', how='left')

    df_merged = df_merged.drop(['DT_SIN_PRI', 'ID_MUNICIP'], axis=1)
    return df_merged

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
        # DT_SIN_PRI signigica a data dos primeiros sintomas + importante do que a data em que a pessoa foi no médico!
        df = pd.read_csv(arquivo, encoding='utf-8', low_memory=False)
        df['DT_SIN_PRI'] = pd.to_datetime(df['DT_SIN_PRI'], format='%Y%m%d', errors='coerce')
        df['ID_MUNICIP'] = df['ID_MUNICIP'].astype(str)
        df = df[['DT_SIN_PRI', 'ID_MUNICIP']].groupby(['ID_MUNICIP', 'DT_SIN_PRI']).size().reset_index(name='DENG_CASES')
        
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
        
        # Fazendo um shift para fazer a previsão a partir das variáveis do dia anterior.
        columns_to_shift = [col for col in df.columns if col not in ['Data Medicao', 'Cod IBGE']]
        df[columns_to_shift] = df[columns_to_shift].shift(1)

        dataframes.append(df)

    climate = pd.concat(dataframes, ignore_index=True)

    climate.to_csv('data/processed/climate.csv', sep=';', index=False)
    return climate

def prepare_single_prediction_features(ibge_code, recent_dengue_cases, df_climate_recent):
    df_climate_recent['Cod IBGE'] = ibge_code
    df_climate_recent['DENG_CASES'] = recent_dengue_cases

    df_socio = socio_data()
    df_merged = pd.merge(df_climate_recent, df_socio, left_on='Cod IBGE', right_on='Código IBGE', how='left')

    df_processed = preprocess_data(df_merged)
    return df_processed

def process_data_forecast_horizon(filepath, forecast_horizon=7):
    df = pd.read_csv(filepath, sep=';', parse_dates=['Data_Medicao'], date_format='%Y-%m-%d')
    
    # Cria as features (variáveis de entrada)
    df = create_future_targets(df, horizon=forecast_horizon)
    df = create_lags(df, lags=[2, 4, 7])

    # Cria features baseadas na data
    df['mes'] = df['Data_Medicao'].dt.month
    df['dia_da_semana'] = df['Data_Medicao'].dt.weekday
    df['ano'] = df['Data_Medicao'].dt.year
    df['sin_mes'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['cos_mes'] = np.cos(2 * np.pi * df['mes'] / 12)

    # Remove linhas com NaN que foram criadas pelos lags e future_targets
    df = df.dropna()

    # Define as colunas alvo (y) e as colunas de features (X)
    target_cols = [f'DENG_CASES_+{i}' for i in range(1, forecast_horizon + 1)]
    y = df[target_cols]
    X = df.drop(columns=['DENG_CASES', 'Data_Medicao'] + target_cols)

    # Aplica one-hot encoding nas features categóricas
    X = pd.get_dummies(X, columns=['Cod_IBGE'])

    return X, y

def create_future_targets(df, horizon):
    for i in range(1, horizon + 1):
        df[f'DENG_CASES_+{i}'] = df['DENG_CASES'].shift(-i)
    return df

def create_lags(df, lags):
    for lag in lags:
        df[f'DENG_CASES_lag_{lag}'] = df['DENG_CASES'].shift(lag)
    return df