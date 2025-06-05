"""Random forest model."""
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def predict_deng():
    df = load_and_preprocess_data('data/processed/table_processed.csv')
    
    # Separando os dados e criando param_grid
    X = df.drop(columns=['DENG_CASES'])
    y = df['DENG_CASES']
    y.to_csv('data/processed/deng_cases.csv', sep=';', index=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None]
    }

    y_pred = train_model(X_train, y_train, X_test, y_test, param_grid)
    save_predictions(y_test, y_pred)

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath, sep=';', parse_dates=['Data_Medicao'], date_format='%Y-%m-%d')
    df = create_lags(df, lags=[1, 7, 10, 30, 60])

    # Lidando com os dados de Data
    df['mes'] = df['Data_Medicao'].dt.month
    df['dia_da_semana'] = df['Data_Medicao'].dt.weekday
    df['ano'] = df['Data_Medicao'].dt.year
    df = df.drop(columns=['Data_Medicao'])
    df['sin_mes'] = np.sin(2 * np.pi * df['mes'] / 12)
    df['cos_mes'] = np.cos(2 * np.pi * df['mes'] / 12)

    df = pd.get_dummies(df, columns=['Cod_IBGE'])
    return df

def create_lags(df, lags):
    for lag in lags:
        df[f'DENG_CASES{lag}'] = df['DENG_CASES'].shift(lag)
    return df.dropna()

def train_model(X_train, y_train, X_test, y_test, param_grid):
    gbr = GradientBoostingRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=gbr,
        param_distributions=param_grid,
        n_iter=30,
        scoring='neg_mean_squared_error',
        cv=tscv,
        verbose=1,
        n_jobs=-1,
        random_state=42
    )
    search.fit(X_train, y_train)

    print("Melhores parâmetros encontrados:")
    print(search.best_params_)

    y_pred = gbr.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n➡️  RMSE: {rmse:.2f}")
    print(f"➡️  R² Score: {r2:.2f}")

    # Plotando as variáveis com mais importância
    importances = gbr.feature_importances_
    sorted_idx = importances.argsort()
    plt.figure(figsize=(10, 8))
    plt.barh(X_train.columns[sorted_idx], importances[sorted_idx])
    plt.title("Importância das Variáveis no Modelo Final")
    plt.xlabel("Importância")
    plt.tight_layout()
    plt.show()

    return y_pred

def save_predictions(y_test, y_pred):
    df_resultados = pd.DataFrame({
        'Real': y_test.values,
        'Previsto': y_pred
    })
    df_resultados.to_csv('resultados_previstos.csv', index=False)

# def modelsComparison():
#     modelos = {
#         'Linear Regression': LinearRegression(),
#         'Random Forest': RandomForestRegressor(),
#         'XGBoost': XGBRegressor(),
#         'Gradient Boosting': GradientBoostingRegressor(),
#         'HistGradientBoosting': HistGradientBoostingRegressor()
#     }

#     resultados = []

#     for nome, modelo in modelos.items():
#         modelo.fit(X_train, y_train)
#         y_pred = modelo.predict(X_test)
#         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#         r2 = r2_score(y_test, y_pred)
#         resultados.append({
#             'Modelo': nome,
#             'RMSE': round(rmse, 2),
#             'R²': round(r2, 2)
#         })

#     resultados_df = pd.DataFrame(resultados).sort_values(by='R²', ascending=False)
#     print(resultados_df)

#     modelo = modelos['Gradient Boosting']

#     coeficientes = pd.DataFrame({
#         'Variavel': X.columns,
#         'Coeficiente': modelo.coef_
#     }).sort_values(by='Coeficiente', key=abs, ascending=False)

#     print(coeficientes.head(10))

#     plt.figure(figsize=(10,6))
#     plt.barh(coeficientes['Variavel'][:15][::-1], coeficientes['Coeficiente'][:15][::-1])
#     plt.xlabel('Peso (coef.)')
#     plt.title('Importância das variáveis (Regressão Linear)')
#     plt.tight_layout()
#     plt.show()
