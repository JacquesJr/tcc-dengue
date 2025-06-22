from flask import Flask
import joblib
import pandas as pd
from flask import request, jsonify, render_template

from src import prepare_single_prediction_features

app = Flask(__name__)

model = None
scaler = None
training_columns = None

def load_resources():
    global model, scaler, training_columns
    model_path = 'app/Lasso_model.pkl'
    scaler_path = 'app/scaler.pkl'
    columns_path = 'app/training_columns.pkl'

    try:
        model = joblib.load(model_path)
        print(f"Modelo carregado com sucesso de '{model_path}'!")
    except Exception as e:
        print(f"Erro ao carregar ou criar modelo: {e}")
        model = None

    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler carregado com sucesso de '{scaler_path}'!")
    except Exception as e:
        print(f"Erro ao carregar ou criar scaler: {e}")
        scaler = None
    
    try:
        training_columns = joblib.load(columns_path)
        print(f"Colunas de treinamento carregadas com sucesso de '{columns_path}'!")
    except Exception as e:
        print(f"Erro ao carregar colunas de treinamento: {e}")
        training_columns = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or training_columns is None:
        return jsonify({'error': 'Recursos do modelo não carregados. Verifique os logs do servidor.'}), 500

    try:
        # Coleta os dados do formulário
        ibge_code = request.form.get('ibge_code')
        if not ibge_code:
            return jsonify({'error': 'Código IBGE é obrigatório.'}), 400

        climate_df = pd.read_csv(request.form.get('csv_file'), sep=';')
        recent_dengue_cases = []
        for i in range(1, 8):
            case_value = request.form.get(f'dengue_d_minus_{i}')
            if case_value is None or case_value == '':
                 return jsonify({'error': f'O valor de casos de Dengue para D-{i} é obrigatório.'}), 400
            recent_dengue_cases.append(int(case_value))

        X_pred_raw = prepare_single_prediction_features(ibge_code, recent_dengue_cases, climate_df)
        X_pred_scaled = scaler.transform(X_pred_raw)

        # Realiza a predição
        prediction = model.predict(X_pred_scaled)
        output = prediction[0].tolist()

        return jsonify({'prediction_7_days': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    load_resources()
    app.run(debug=True)
