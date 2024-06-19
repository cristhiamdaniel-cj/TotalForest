# API para Predicción con Flask

## Descripción General
Esta API basada en Flask permite realizar predicciones utilizando un modelo de Random Forest previamente entrenado. Los usuarios pueden enviar datos a la API y recibir una predicción en tiempo real.

## Estructura del Proyecto
El proyecto incluye los siguientes archivos clave:
- `app.py`: El servidor Flask que maneja las solicitudes de predicción.
- `index.html`: La página web principal para interactuar con la API.
- `script.js`: El archivo JavaScript que maneja la lógica del formulario.
- `style.css`: El archivo CSS que define el estilo de la página web.

## Código

### `api.py`
Este archivo define la lógica del servidor Flask y los endpoints para la predicción.

```python
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
```

### `index.html`
La interfaz principal de la aplicación donde los usuarios pueden ingresar datos para la predicción.

```html
<!DOCTYPE html>
<html>
<head>
    <title>Model Prediction API</title>
    <link rel="stylesheet" type="text/css" href="{{ url_for('static', filename='style.css') }}">
</head>
<body>
    <div class="container">
        <header>
            <img src="https://ekosnegocios.com/image/posts/header/9256.jpg" alt="Header Image">
            <h1>Model Prediction API</h1>
        </header>
    <form id="prediction-form">
        <label for="Unnamed: 0">Unnamed: 0:</label>
        <input type="text" id="Unnamed: 0" name="Unnamed: 0" required><br><br>
        <label for="Age">Age:</label>
        <input type="text" id="Age" name="Age" required><br><br>
        <label for="Sex_male">Sex (male=1, female=0):</label>
        <input type="text" id="Sex_male" name="Sex_male" required><br><br>
        <label for="Job">Job:</label>
        <input type="text" id="Job" name="Job" required><br><br>
        <label for="Housing_own">Housing (own=1, rent=0):</label>
        <input type="text" id="Housing_own" name="Housing_own" required><br><br>
        <label for="Saving_accounts_little">Saving accounts (little=1, moderate=0.5, rich=0):</label>
        <input type="text" id="Saving_accounts_little" name="Saving_accounts_little" required><br><br>
        <label for="Checking_account_moderate">Checking account (moderate=1, little=0.5, rich=0):</label>
        <input type="text" id="Checking_account_moderate" name="Checking_account_moderate" required><br><br>
        <label for="Credit_amount">Credit amount:</label>
        <input type="text" id="Credit_amount" name="Credit_amount" required><br><br>
        <label for="Duration">Duration:</label>
        <input type="text" id="Duration" name="Duration" required><br><br>
        <label for="Purpose_vacation/others">Purpose (vacation/others=1, repairs=0):</label>
        <input type="text" id="Purpose_vacation/others" name="Purpose_vacation/others" required><br><br>
        <label for="Purpose_repairs">Purpose (repairs=1, vacation/others=0):</label>
        <input type="text" id="Purpose_repairs" name="Purpose_repairs" required><br><br>

        <!-- Añadir todos los campos faltantes con un valor predeterminado -->
        <input type="hidden" name="Sex_female" value="0">
        <input type="hidden" name="Housing_rent" value="0">
        <input type="hidden" name="Housing_free" value="0">
        <input type="hidden" name="Saving_accounts_moderate" value="0">
        <input type="hidden" name="Saving_accounts_rich" value="0">
        <input type="hidden" name="Saving_accounts_nan" value="0">
        <input type="hidden" name="Checking_account_little" value="0">
        <input type="hidden" name="Checking_account_rich" value="0">
        <input type="hidden" name="Checking_account_nan" value="0">
        <input type="hidden" name="Purpose_radio/TV" value="0">
        <input type="hidden" name="Purpose_furniture/equipment" value="0">
        <input type="hidden" name="Purpose_new car" value="0">
        <input type="hidden" name="Purpose_used car" value="0">
        <input type="hidden" name="Purpose_business" value="0">
        <input type="hidden" name="Purpose_domestic appliances" value="0">
        <input type="hidden" name="Purpose_education" value="0">
        <input type="hidden" name="Purpose_car" value="0">
        <input type="hidden" name="Saving_accounts_quite rich" value="0">

        <input type="submit" value="Predict">
    </form>
    <h2 id="result"></h2>
    <h3 id="category"></h3>
    <script src="{{ url_for('static', filename='script.js') }}"></script>
</body>
</html>
```

### `script.js`
El archivo JavaScript maneja la lógica del formulario y la comunicación con la API.

```javascript
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('result');
    const categoryDiv = document.getElementById('category');

    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        const data = {};

        formData.forEach((value, key) => {
            data[key] = parseFloat(value) || value;
        });

        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            ```javascript
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                resultDiv.innerHTML = `Error: ${data.error}`;
                categoryDiv.innerHTML = '';
            } else {
                resultDiv.innerHTML = `Prediction: ${data.prediction.toFixed(2)}`;
                categoryDiv.innerHTML = `Category: ${data.prediction_category}`;
            }
        })
        .catch(error => {
            console.error('Error:', error);
            resultDiv.innerHTML = 'Error al realizar la predicción';
            categoryDiv.innerHTML = '';
        });
    });
});
```

### `style.css`
El archivo CSS define el estilo de la página web para que se vea atractiva y profesional.

```css
body {
    font-family: Arial, sans-serif;
    background-color: #f4f4f4;
    margin: 0;
    padding: 0;
}

.container {
    width: 50%;
    margin: 0 auto;
    background-color: #ffffff;
    padding: 20px;
    box-shadow: 0 0 10px rgba(0,0,0,0.1);
    border-radius: 8px;
    margin-top: 30px;
}

header {
    text-align: center;
}

header img {
    width: 100%;
    border-radius: 8px 8px 0 0;
}

h1 {
    color: #333333;
}

form {
    margin-top: 20px;
}

.form-group {
    margin-bottom: 15px;
}

.form-group label {
    display: block;
    color: #666666;
    margin-bottom: 5px;
}

.form-group input[type="text"] {
    width: 100%;
    padding: 8px;
    border: 1px solid #cccccc;
    border-radius: 4px;
    box-sizing: border-box;
}

.form-group input[type="submit"] {
    width: 100%;
    padding: 10px;
    background-color: #4CAF50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}

.form-group input[type="submit"]:hover {
    background-color: #45a049;
}

#result, #category {
    margin-top: 20px;
    padding: 10px;
    background-color: #e7f3e7;
    border: 1px solid #c3e6c3;
    border-radius: 4px;
    color: #333333;
    font-weight: bold;
}
```

## Cómo Ejecutar la Aplicación

### Requisitos
1. Python 3.9 o superior
2. Flask
3. joblib

### Pasos para Ejecutar
1. Clona el repositorio y navega al directorio del proyecto.
2. Asegúrate de tener todas las dependencias instaladas. Puedes instalarlas utilizando el siguiente comando:
    ```bash
    pip install -r requirements.txt
    ```
3. Navega al directorio `src` y ejecuta el archivo `app.py`:
    ```bash
    cd src
    python app.py
    ```
4. Abre tu navegador web y navega a `http://127.0.0.1:5000` para acceder a la aplicación.

### Uso
1. Ingresa los datos en el formulario presentado en la página principal.
2. Haz clic en el botón "Predict" para obtener la predicción.
3. La predicción y la categoría se mostrarán en la parte inferior de la página.

## Conclusión
Esta API de Flask permite a los usuarios realizar predicciones en tiempo real utilizando un modelo de Random Forest. La interfaz web facilita la interacción con la API y presenta los resultados de manera clara y concisa. Esta estructura modular permite una fácil extensión y mantenimiento del código.

El resultado de la página: ![model_api_1.png](model_api.png)