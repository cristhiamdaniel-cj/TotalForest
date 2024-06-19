# TotalForest

## Descripción

TotalForest es una aplicación para la predicción de riesgos basada en modelos de aprendizaje automático. Utiliza un modelo de Random Forest entrenado para hacer predicciones a partir de datos financieros.

## Requisitos

Asegúrate de tener instalados los siguientes componentes en tu sistema:

- Python 3.9 o superior
- Pip (el gestor de paquetes de Python)
- Virtualenv (opcional pero recomendado)

## Instrucciones de instalación

1. Clona el repositorio en tu máquina local:

    ```bash
    git clone https://github.com/tuusuario/TotalReport.git
    cd TotalReport
    ```

2. Crea un entorno virtual (opcional pero recomendado):

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
    ```

3. Instala las dependencias necesarias:

    ```bash
    pip install -r requirements.txt
    ```

## Preparar los datos

Asegúrate de que los datos necesarios están en la carpeta `data/raw/`. Los datos utilizados deben estar en formato CSV y seguir la estructura esperada por los scripts de entrenamiento y predicción.

## Entrenar el modelo

Si deseas entrenar el modelo, ejecuta el siguiente script. Este paso no es necesario si ya tienes un modelo preentrenado en la carpeta `output`.

```bash
python src/neural_network.py
```

## Ejecutar la API

Para iniciar la API de Flask, navega hasta el directorio `api` y ejecuta el script `api.py`.

```bash
cd api
python api.py
```

La API estará disponible en `http://127.0.0.1:5000`.

## Estructura del proyecto

```
TotalReport/
├── api/
│   ├── api.py
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   └── templates/
│       └── index.html
├── data/
│   └── raw/
│       └── german_credit_data.csv
├── output/
│   ├── boxplot_numerico.png
│   ├── DecisionTreeRegressor.joblib
│   ├── distribucion_categorica.png
│   ├── distribucion_riesgo.png
│   ├── matriz_correlacion.png
│   ├── model_comparison_results.csv
│   ├── neural_network_model.h5
│   ├── RandomForestRegressor.joblib
│   ├── scaled_data.csv
│   ├── test_data.csv
│   └── train_data.csv
├── src/
│   ├── app.py
│   ├── data_procesador.py
│   ├── feature_engineering.py
│   ├── logger.py
│   ├── modeling.py
│   ├── neural_network.py
│   └── templates/
│       └── ...
├── Dockerfile (opcional)
├── docker-compose.yml (opcional)
├── README.md
└── requirements.txt
```

## Uso de la API

1. Abre un navegador web y navega a `http://127.0.0.1:5000`.
2. Introduce los datos requeridos en el formulario.
3. Haz clic en el botón "Predict" para obtener la predicción.

