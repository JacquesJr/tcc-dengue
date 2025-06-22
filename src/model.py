"""Linear models for dengue prediction."""
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from .data_processing import process_data_forecast_horizon

import matplotlib.pyplot as plt
import pandas as pd
import joblib

def predict_deng():
    # Alterado para prever apenas 1 dia no futuro
    FORECAST_HORIZON = 1
    # Carrega e pré-processa os dados, recebendo os dataframes X e y
    X, y = process_data_forecast_horizon('data/processed/table_processed.csv', forecast_horizon=FORECAST_HORIZON)

    # Separa os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Padronização das Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    y_pred, model_name, important_features = train_model(X_train, y_train, X_test, y_test, scaler)
    save_predictions(y_test, y_pred, model_name, forecast_horizon=FORECAST_HORIZON)

    # Salva a lista de variáveis importantes para uso futuro
    important_features_path = 'src/app/important_variables.pkl'
    joblib.dump(important_features, important_features_path)
    print(f"\nVariáveis importantes salvas em '{important_features_path}'")

def train_model(X_train, y_train, X_test, y_test, scaler_obj):
    model_name = 'Lasso'
    model = Lasso(fit_intercept=True, alpha=0.01, random_state=42, max_iter=100000)

    print(f"\n--- Treinando o modelo: {model_name} ---")
    print(f"Usando parâmetros: {{'fit_intercept': True, 'alpha': 0.01, 'max_iter': 100000}}")
    
    model.fit(X_train, y_train)
    print(f"Modelo '{model_name}' treinado com sucesso.")

    model_path = f'src/app/{model_name}_model.pkl'
    scaler_path = f'src/app/scaler.pkl'
    columns_path = f'src/app/training_columns.pkl'

    joblib.dump(model, model_path)
    print(f"Modelo '{model_name}' salvo em '{model_path}'")

    joblib.dump(scaler_obj, scaler_path)
    print(f"Scaler salvo em '{scaler_path}'")

    joblib.dump(X_train.columns.tolist(), columns_path)
    print(f"Colunas de treinamento salvas em '{columns_path}'")

    y_pred = model.predict(X_test)

    # Calcula e exibe as métricas de avaliação
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n--- Métricas de Avaliação ({model_name}) ---")
    print(f"➡️  RMSE no conjunto de teste: {rmse:.2f}")
    print(f"➡️  R² Score no conjunto de teste: {r2:.2f}")

    # Variáveis mais importantes (Lasso zera as variáveis menos importantes)
    coefficients = model.coef_.ravel()
    feature_importance = pd.Series(coefficients, index=X_train.columns)
    important_features = feature_importance[feature_importance != 0]

    print("\n--- Variáveis Mais Impactantes (selecionadas pelo Lasso) ---")
    print(important_features.abs().sort_values(ascending=False))

    important_feature_names = important_features.index.tolist()

    return y_pred, model_name, important_feature_names

def save_predictions(y_test, y_pred, model_name, forecast_horizon=7):
    pred_cols = [f'Previsto_D+{i}' for i in range(1, forecast_horizon + 1)]
    df_pred = pd.DataFrame(y_pred, columns=pred_cols, index=y_test.index)

    # Concatena os valores reais e previstos
    df_resultados = pd.concat([y_test, df_pred], axis=1)
    df_resultados.to_csv(f'data/results/resultados_previstos_{model_name}.csv', index=False)

    # Ajuste para o plot funcionar com 1 ou mais dias de previsão
    real_values = y_test.iloc[:, 0].values
    if y_pred.ndim > 1:
        predicted_values = y_pred[:, 0]
    else:
        predicted_values = y_pred

    plt.figure(figsize=(12, 7))
    plt.plot(real_values, label='Real (D+1)', color='blue', marker='o', linestyle='-', markersize=4)
    plt.plot(predicted_values, label='Previsto (D+1)', color='red', marker='x', linestyle='--', markersize=4)
    plt.title(f'Comparação Real vs. Previsto para 1 Dia à Frente ({model_name})')
    plt.xlabel('Amostras no Conjunto de Teste')
    plt.ylabel('Casos de Dengue')
    plt.legend()
    plt.savefig(f'data/results/real_vs_previsto_{model_name}_D1.png')
    plt.close()