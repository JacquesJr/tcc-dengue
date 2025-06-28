from flask import Flask
import joblib
import pandas as pd
from flask import request, jsonify, render_template
from pathlib import Path

from src import prepare_single_prediction_features

app = Flask(__name__)

# Define a base directory for artifacts relative to this file's location
BASE_DIR = Path(__file__).resolve().parent
model = None
scaler = None
training_columns = None

def load_resources():
    global model, scaler, training_columns
    # Use the robust paths based on the app's location
    model_path = BASE_DIR / 'RandomForest_model.pkl'
    scaler_path = BASE_DIR / 'scaler.pkl'
    columns_path = BASE_DIR / 'training_columns.pkl'

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

# Load resources when the application starts, making it ready for production servers
load_resources()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None or training_columns is None:
        return jsonify({'error': 'Recursos do modelo não carregados. Verifique os logs do servidor.'}), 500

    try:
        ibge_code = request.form.get('ibge_code')
        if not ibge_code: # Check for None or empty string
            return jsonify({'error': 'Código IBGE é obrigatório.'}), 400
        try:
            ibge_code = int(ibge_code) // 10
        except ValueError:
            return jsonify({'error': 'Código IBGE inválido. Deve ser um número inteiro.'}), 400

        # Correctly handle the file upload from request.files
        if 'csv_file' not in request.files:
            return jsonify({'error': 'Nenhum arquivo CSV foi enviado.'}), 400
        
        csv_file = request.files['csv_file']
        if csv_file.filename == '': # Check if a file was actually selected
            return jsonify({'error': 'Nenhum arquivo CSV foi selecionado.'}), 400

        try:
            climate_df = pd.read_csv(csv_file, sep=';')
        except pd.errors.EmptyDataError:
            return jsonify({'error': 'O arquivo CSV está vazio.'}), 400
        except pd.errors.ParserError as e:
            return jsonify({'error': f'Erro ao analisar o arquivo CSV: {e}. Verifique o formato e o separador (;).'}), 400
        except Exception as e: # Catch other potential issues with reading CSV
            return jsonify({'error': f'Erro inesperado ao ler o arquivo CSV: {e}.'}), 400

        recent_dengue_cases = []
        for i in range(1, 8):
            case_value = request.form.get(f'dengue_d_minus_{i}')
            if case_value is None or case_value == '':
                 return jsonify({'error': f'O valor de casos de Dengue para D-{i} é obrigatório.'}), 400
            try:
                recent_dengue_cases.append(int(case_value))
            except ValueError:
                return jsonify({'error': f'Valor inválido para casos de Dengue D-{i}. Deve ser um número inteiro.'}), 400

        X_pred_raw = prepare_single_prediction_features(ibge_code, recent_dengue_cases, climate_df)

        print(X_pred_raw)

        X_pred_scaled = scaler.transform(X_pred_raw)
        prediction = model.predict(X_pred_scaled)
        output = prediction[0].tolist()

        return jsonify({'prediction_dengue_cases': output})
    except Exception as e:
        app.logger.error(f"Prediction failed: {e}")
        return jsonify({'error': f'Ocorreu um erro inesperado no servidor: {e}'}), 500

if __name__ == '__main__':
    # The app.run() is only for local development.
    app.run(debug=True)
