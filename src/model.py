"""Linear models for dengue prediction."""
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
from .data_processing import process_data

import matplotlib.pyplot as plt
import pandas as pd
import joblib

def predict_deng():
    X, y = process_data('data/processed/table_processed.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Padronização das Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    y_pred, model_name, important_features = train_model(X_train, y_train, X_test, y_test, scaler)
    save_predictions(y_test, y_pred, model_name)

    important_features_path = 'src/app/important_variables.pkl'
    joblib.dump(important_features, important_features_path)
    print(f"\nVariáveis importantes salvas em '{important_features_path}'")

def train_model(X_train, y_train, X_test, y_test, scaler_obj):
    model_name = 'RandomForest'
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    print(f"\n--- Treinando o modelo: {model_name} ---")
    print(f"Usando parâmetros: {{'n_estimators': 100, 'random_state': 42, 'n_jobs': -1}}")
    
    model.fit(X_train, y_train.values.ravel())
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

    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n--- Métricas de Avaliação ({model_name}) ---")
    print(f"➡️  RMSE no conjunto de teste: {rmse:.2f}")
    print(f"➡️  R² Score no conjunto de teste: {r2:.2f}")

    feature_importances = model.feature_importances_
    important_features = pd.Series(feature_importances, index=X_train.columns).sort_values(ascending=False)

    print("\n--- Variáveis Mais Impactantes (Feature Importances) ---")
    print(important_features)

    important_feature_names = important_features.index.tolist()

    return y_pred, model_name, important_feature_names

def save_predictions(y_test, y_pred, model_name):
    pred_cols = ['Previsto_D+1']
    df_pred = pd.DataFrame(y_pred, columns=pred_cols, index=y_test.index)

    df_resultados = pd.concat([y_test, df_pred], axis=1)
    df_resultados.to_csv(f'data/results/resultados_previstos_{model_name}.csv', index=False)
    real_values = y_test.iloc[:, 0].values
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