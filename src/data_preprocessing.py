"""Import panda to read csv."""
import os
import pandas as pd

def load_data():
    """Read data from csv and merge all dengue tables in one."""
    city_data = pd.read_csv("data/raw/ips_brasil_municipios.csv", sep=',', encoding='utf-8')

    # Load dengue_cases databases
    dengue_cases_22 = pd.read_csv("data/raw/DENGBR22.csv", sep=',', encoding='utf-8',
                                low_memory=False, quotechar='"')

    return dengue_cases_22, city_data

def preprocess_and_merge(table_dengue, table_city):
    """Function to preprocess and merge tables."""

    #Preprocessing table_dengue
    table_dengue["ID_UNIDADE"] = table_dengue["ID_UNIDADE"].fillna(0).astype(int)

    columns_keep = ["DT_NOTIFIC", "NU_ANO", "SG_UF_NOT", "ID_MUNICIP", "ID_UNIDADE", "DT_SIN_PRI"]
    table_dengue = table_dengue[columns_keep]

    os.makedirs("data/processed", exist_ok=True)
    table_dengue.to_csv("data/processed/dengue_processed.csv")

    merged_table = table_dengue.groupby('ID_MUNICIP').size().reset_index(name='DENG_CASES_COUNT')

    merged = pd.merge(table_city, merged_table, left_on='Código IBGE', right_on='ID_MUNICIP',
                      how='left')
    merged['DENG_CASES_COUNT'] = merged['DENG_CASES_COUNT'].fillna(0)

    os.makedirs("data/processed", exist_ok=True)
    table_dengue.to_csv("data/processed/dengue_cases_count.csv")

    return merged

def prepocess_uberaba_data():
    dengue22 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR22.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue23 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR23.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue24 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR24.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue_merged = pd.concat([dengue22, dengue23, dengue24], ignore_index=True)

    uberaba_socio = preprocess_uberaba_socio(pd.read_csv('data/raw/ips_brasil_municipios.csv'))

    climate_uberaba22 = pd.read_csv('data/raw/Uberaba22.csv', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')
    climate_uberaba23 = pd.read_csv('data/raw/Uberaba23.csv', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')
    climate_uberaba24 = pd.read_csv('data/raw/Uberaba24.csv', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')
    climate_uberaba_merged = pd.concat([climate_uberaba22, climate_uberaba23, climate_uberaba24], ignore_index=True)

    # Merge com clima
    merged = pd.merge(climate_uberaba_merged, dengue_merged, left_on='Data Medicao', right_on='DT_NOTIFIC', how='right')
    merged['DENG_CASES'] = merged['DENG_CASES'].fillna(0)

    # Pegar apenas as informações socioeconômicas de Uberaba
    for col in uberaba_socio.columns:
        if col != 'Cidade':
            merged[col] = uberaba_socio[col].values[0] 

    # Criar variáveis de "lag" (histórico de casos passados)
    for lag in range(1, 8):  # últimos 7 dias
        merged[f'lag_{lag}'] = merged['DENG_CASES'].shift(lag)

    merged = merged.dropna()

    return merged

def preprocess_uberaba_deng(table):
    #Tratando a tabela de casos de dengue
    table["ID_UNIDADE"] = table["ID_UNIDADE"].fillna(0).astype(int)

    columns_keep = ["DT_NOTIFIC", "NU_ANO", "SG_UF_NOT", "ID_MUNICIP", "ID_UNIDADE", "DT_SIN_PRI"]
    table = table[columns_keep]

    #Extraindo apenas casos de uberaba
    uberaba_dengue_daily = table[table['ID_MUNICIP'] == 317010]
    uberaba_dengue_daily = uberaba_dengue_daily.groupby('DT_NOTIFIC').size().reset_index(name='DENG_CASES')

    #Salvando a tabela
    os.makedirs("data/processed", exist_ok=True)
    uberaba_dengue_daily.to_csv("data/processed/dengue_processed.csv")
    
    return uberaba_dengue_daily

def preprocess_uberaba_socio(table):
    #Preprocessing city table
    table['Código IBGE'] = table['Código IBGE'] // 10
    uberaba_socio = table[table['Código IBGE'] == 317010]

    return uberaba_socio
