import os
from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)
model_path = '../output/RandomForestRegressor.joblib'
model = joblib.load(model_path)
print(f'Modelo cargado desde {model_path}')

# Imprimir las características que espera el modelo
expected_features = model.feature_names_in_
print(f'Características esperadas por el modelo: {expected_features}')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Verificar los datos recibidos
        print(f'Datos recibidos: {request.json}')

        # Recuperar los datos del formulario
        data = request.json

        # Calcular CreditAmountPerMonth
        data['CreditAmountPerMonth'] = data['Credit_amount'] / data['Duration']

        # Crear un array de entrada para el modelo
        input_data = [
            data.get('Unnamed: 0', 0), data.get('Age', 0), data.get('Job', 0), data.get('Credit_amount', 0),
            data.get('Duration', 0),
            data.get('CreditAmountPerMonth', 0), data.get('Sex_female', 0), data.get('Sex_male', 0),
            data.get('Housing_free', 0),
            data.get('Housing_own', 0), data.get('Housing_rent', 0), data.get('Saving_accounts_little', 0),
            data.get('Saving_accounts_moderate', 0), data.get('Saving_accounts_quite rich', 0),
            data.get('Saving_accounts_rich', 0),
            data.get('Saving_accounts_nan', 0), data.get('Checking_account_little', 0),
            data.get('Checking_account_moderate', 0),
            data.get('Checking_account_rich', 0), data.get('Checking_account_nan', 0), data.get('Purpose_business', 0),
            data.get('Purpose_car', 0), data.get('Purpose_domestic appliances', 0), data.get('Purpose_education', 0),
            data.get('Purpose_furniture/equipment', 0), data.get('Purpose_radio/TV', 0), data.get('Purpose_repairs', 0),
            data.get('Purpose_vacation/others', 0)
        ]

        # Imprimir los datos de entrada y las características esperadas por el modelo
        print(f'Datos de entrada: {input_data}')
        print(f'Número de características en los datos de entrada: {len(input_data)}')
        print(f'Número de características esperadas por el modelo: {len(expected_features)}')

        # Realizar la predicción
        prediction = model.predict([input_data])[0]

        # Convertir la predicción a una categoría
        prediction_category = 'Bueno' if prediction >= 0.5 else 'Malo'

        # Devolver la predicción y la categoría como JSON
        return jsonify({
            'prediction': prediction,
            'prediction_category': prediction_category
        })
    except Exception as e:
        print(f'Error al realizar la predicción: {e}')
        return jsonify({'error': str(e)})


if __name__ == "__main__":
    app.run(debug=True)
