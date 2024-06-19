# Aplicación Básica de Visualización de Datos con Flask

## Descripción General
La aplicación `Flask` proporciona una interfaz web para cargar, visualizar y explorar datos de un archivo CSV. Los datos se procesan y se generan visualizaciones que se pueden acceder a través de diferentes endpoints.

## Código

### Importaciones y Configuración Inicial
El código importa las bibliotecas necesarias, define las rutas para los archivos CSV y de salida, y crea instancias del procesador de datos y el logger.

```python
from flask import Flask, request, jsonify, send_file, render_template_string, render_template
import os
from io import StringIO
from src.data_procesador import DataProcessor
from src.logger import CustomLogger

app = Flask(__name__)

# Ruta para el archivo CSV y la carpeta de salida
CSV_PATH = os.path.join(os.path.dirname(__file__), '../data/raw/german_credit_data.csv')
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '../output')

# Instanciar el procesador de datos y el logger
logger = CustomLogger().get_logger()
data_processor = DataProcessor(CSV_PATH, OUTPUT_PATH)

# Cargar y procesar los datos al iniciar la aplicación
data_processor.cargar_datos()
data_processor.limpiar_datos()
data_processor.analisis_exploratorio_datos()
```

### Página de Inicio
La página de inicio ofrece enlaces a varias visualizaciones y funciones de datos.

```python
# Página de inicio con enlaces a las visualizaciones
@app.route('/')
def index():
    html = '''
    <h1>Bienvenido a la Aplicación de Visualización de Datos</h1>
    <ul>
        <li><a href="/load_data">Cargar Datos</a> </li>
        <li><a href="/info_data">Información del Dataset</a> </li>
        <li><a href="/describe_data">Descripción del Dataset</a> </li>
        <li><a href="/plot/boxplot_numerico">Boxplot Numérico</a> </li>
        <li><a href="/plot/distribucion_categorica">Distribución Categórica</a> </li>
        <li><a href="/plot/distribucion_numerica">Distribución Numérica</a> </li>
        <li><a href="/plot/distribucion_riesgo">Distribución de Riesgo</a> </li>
        <li><a href="/plot/matriz_correlacion">Matriz de Correlación</a> </li>
    </ul>
    '''
    return render_template_string(html)
```

### Endpoints para Visualización de Datos
Los siguientes endpoints permiten a los usuarios cargar y visualizar datos y gráficos generados a partir del análisis exploratorio.

#### Cargar Datos
Muestra los primeros registros del CSV cargado.

```python
@app.route('/load_data', methods=['GET'])
def load_data_route():
    df = data_processor.data
    data = df.head().to_dict(orient='records')
    columns = df.columns.tolist()
    return render_template('data.html', data=data, columns=columns)
```

#### Información del Dataset
Muestra información general sobre el dataset, como el tipo de datos y los valores no nulos.

```python
@app.route('/info_data', methods=['GET'])
def info_data_route():
    buffer = StringIO()
    data_processor.data.info(buf=buffer)
    info_str = buffer.getvalue()
    return render_template_string('<pre>{{ info_str }}</pre>', info_str=info_str)
```

#### Descripción del Dataset
Proporciona estadísticas descriptivas del dataset.

```python
@app.route('/describe_data', methods=['GET'])
def describe_data_route():
    describe_str = data_processor.data.describe().to_html()
    return render_template_string('<div>{{ describe_str|safe }}</div>', describe_str=describe_str)
```

### Función de Ayuda para Servir y Limpiar Archivos de Imagen
Una función auxiliar para servir archivos de imagen y eliminarlos después de servir.

```python
def serve_and_cleanup(image_path):
    """
    Helper function to serve an image file and delete it after serving.
    """
    response = send_file(image_path, mimetype='image/png')
    os.remove(image_path)
    return response
```

### Endpoints para Visualizaciones
Cada uno de estos endpoints sirve una imagen de los gráficos generados.

```python
# Endpoint para mostrar el boxplot numérico
@app.route('/plot/boxplot_numerico', methods=['GET'])
def plot_boxplot_numerico():
    return serve_and_cleanup(os.path.join(OUTPUT_PATH, 'boxplot_numerico.png'))

# Endpoint para mostrar la distribución categórica
@app.route('/plot/distribucion_categorica', methods=['GET'])
def plot_distribucion_categorica():
    return serve_and_cleanup(os.path.join(OUTPUT_PATH, 'distribucion_categorica.png'))

# Endpoint para mostrar la distribución numérica
@app.route('/plot/distribucion_numerica', methods=['GET'])
def plot_distribucion_numerica():
    return serve_and_cleanup(os.path.join(OUTPUT_PATH, 'distribucion_numerica.png'))

# Endpoint para mostrar la distribución de riesgo
@app.route('/plot/distribucion_riesgo', methods=['GET'])
def plot_distribucion_riesgo():
    return serve_and_cleanup(os.path.join(OUTPUT_PATH, 'distribucion_riesgo.png'))

# Endpoint para mostrar la matriz de correlación
@app.route('/plot/matriz_correlacion', methods=['GET'])
def plot_matriz_correlacion():
    return serve_and_cleanup(os.path.join(OUTPUT_PATH, 'matriz_correlacion.png'))
```

### Ejecutar la Aplicación
El siguiente bloque de código ejecuta la aplicación Flask.

```python
if __name__ == "__main__":
    app.run(debug=True)
```

## Templates
### Template para Mostrar Datos (`data.html`)
El template `data.html` muestra los datos del dataset en una tabla HTML.

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Datos</title>
</head>
<body>
    <table border="1">
        <thead>
            <tr>
                {% for column in columns %}
                    <th>{{ column }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
                <tr>
                    {% for key, value in row.items() %}
                        <td>{{ value }}</td>
                    {% endfor %}
                </tr>
            {% endfor %}
        </tbody>
    </table>
</body>
</html>
```

## Conclusión
Esta aplicación Flask permite cargar, limpiar, analizar y visualizar datos de un archivo CSV. Ofrece múltiples endpoints para explorar diferentes aspectos del dataset y proporciona una interfaz web amigable para interactuar con los datos y las visualizaciones generadas.
```