"""Import panda to read csv."""
import os
import pandas as pd

def prepocess_uberaba_data():
    dengue22 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR22.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue23 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR23.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue24 = preprocess_uberaba_deng(pd.read_csv('data/raw/DENGBR24.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False))
    dengue_merged = pd.concat([dengue22, dengue23, dengue24], ignore_index=True)
    uberaba_socio = preprocess_uberaba_socio(pd.read_csv('data/raw/ips_brasil_municipios.csv'))
    climate_uberaba = pd.read_csv('data/raw/Uberaba22-23-24.csv', sep=';', parse_dates=['Data Medicao'], date_format='%Y-%m-%d')

    # Merge com clima
    merged = pd.merge(climate_uberaba, dengue_merged, left_on='Data Medicao', right_on='DT_NOTIFIC', how='right')
    merged['DENG_CASES'] = merged['DENG_CASES'].fillna(0)
    
    print("dengue22:", dengue22.shape)
    print("dengue23:", dengue23.shape)
    print("dengue24:", dengue24.shape)
    print("dengue_merged:", dengue_merged.shape)
    print("merged:", merged.shape)

    # Pegar apenas as informações socioeconômicas de Uberaba
    for col in uberaba_socio.columns:
        if col != 'Cidade':
            merged[col] = uberaba_socio[col].values[0] 

    # Criar variáveis de "lag" (histórico de casos passados)
    for lag in range(1, 8):  # últimos 7 dias
        merged[f'lag_{lag}'] = merged['DENG_CASES'].shift(lag)

    return merged.dropna()

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

def prepocess_data():
    dengue22 = pd.read_csv('data/raw/DENGBR22.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False)
    dengue23 = pd.read_csv('data/raw/DENGBR23.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False)
    dengue24 = pd.read_csv('data/raw/DENGBR24.csv', parse_dates=['DT_NOTIFIC'], date_format='%Y%m%d',
                                                 encoding='utf-8', low_memory=False)
    dengue_merged = pd.concat([dengue22, dengue23, dengue24], ignore_index=True)
    cities_socio = preprocess_socio(pd.read_csv('data/raw/ips_brasil_municipios.csv'))

    dengue_by_city = dengue_merged.groupby('ID_MUNICIP').size().reset_index(name='DENG_CASES')

    # Pegar as informações socioeconômicas
    merged = pd.merge(dengue_by_city, cities_socio, left_on='ID_MUNICIP', right_on='Código IBGE', how='left')

    return merged.dropna()

def preprocess_socio(table):
    #Preprocessing city table
    table['Código IBGE'] = table['Código IBGE'] // 10

    return table