# Data Processor

## Descripción General

La clase `DataProcessor` se encarga de cargar, limpiar y realizar análisis exploratorio de datos (EDA) sobre un conjunto de datos de crédito. El objetivo es transformar y visualizar los datos para extraer información útil y preparar los datos para su posterior modelado.

## Código

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from src.logger import CustomLogger
from prettytable import PrettyTable
import numpy as np
import warnings

class DataProcessor:
    def __init__(self, csv_path, output_path):
        self.csv_path = csv_path
        self.output_path = output_path
        self.logger = CustomLogger().get_logger()
        self.data = None

    def cargar_datos(self):
        try:
            self.data = pd.read_csv(self.csv_path)
            self.logger.info('Datos cargados desde el archivo CSV.')
        except Exception as e:
            self.logger.error(f'Error al cargar los datos: {e}')
            raise

    def limpiar_datos(self):
        try:
            # Llenar valores nulos
            self.data = self.data.fillna('unknown')

            # Convertir tipos de datos
            object_cols = self.data.select_dtypes(include=['object']).columns
            self.data[object_cols] = self.data[object_cols].astype('category')
            self.data['Job'] = self.data['Job'].astype('category')

            self.logger.info('Limpieza de datos completada.')
        except Exception as e:
            self.logger.error(f'Error al limpiar los datos: {e}')
            raise

    def analisis_exploratorio_datos(self):
        try:
            self.logger.info('Iniciando análisis exploratorio de datos.')

            # Suprimir advertencias
            warnings.simplefilter(action='ignore', category=FutureWarning)
            warnings.simplefilter(action='ignore', category=UserWarning)

            # Análisis univariado
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            sns.countplot(x='Risk', data=self.data, ax=ax[0], palette='rocket')
            ax[0].set_title('Valores absolutos de Riesgo')
            colors = sns.color_palette('rocket', n_colors=self.data['Risk'].nunique())
            risk_counts = self.data['Risk'].value_counts()
            ax[1].pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, colors=colors)
            ax[1].add_artist(plt.Circle((0, 0), 0.70, fc='white'))
            ax[1].set_title('Valores relativos de Riesgo')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'distribucion_riesgo.png'))
            plt.close()

            # Análisis numérico
            num_features = ['Age', 'Credit amount', 'Duration']
            fig, ax = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
            ax = ax.flatten()
            for idx, column in enumerate(num_features):
                sns.histplot(data=self.data, x=column, ax=ax[idx], palette='rocket', kde=True)
                ax[idx].set_title(f'Distribución de {column}', size=14)
                ax[idx].set_xlabel(None)
                plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'distribucion_numerica.png'))
            plt.close()

            # Boxplots
            fig, ax = plt.subplots(1, 3, figsize=(10, 5), dpi=300)
            ax = ax.flatten()
            for idx, column in enumerate(num_features):
                sns.boxplot(data=self.data, x=column, ax=ax[idx], palette='rocket')
                ax[idx].set_title(f'Distribución de {column}', size=14)
                ax[idx].set_xlabel(None)
                plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'boxplot_numerico.png'))
            plt.close()

            # Análisis categórico
            cat_features = self.data.select_dtypes(include=['category']).columns.tolist()
            fig, ax = plt.subplots(len(cat_features), 1, figsize=(15, 15), dpi=300)
            ax = ax.flatten()
            for idx, column in enumerate(cat_features):
                sns.countplot(data=self.data, y=column, ax=ax[idx], palette='rocket')
                ax[idx].set_title(f'Distribución de {column}', size=14)
                ax[idx].set_xlabel(None)
                plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'distribucion_categorica.png'))
            plt.close()

            # Codificación de variables categóricas
            self.data = pd.get_dummies(self.data, drop_first=True)

            # Matriz de correlación
            corr_matrix = self.data.corr()
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
            plt.title('Matriz de Correlación')
            plt.savefig(os.path.join(self.output_path, 'matriz_correlacion.png'))
            plt.close()

            self.logger.info('Análisis exploratorio de datos completado y guardado.')
        except Exception as e:
            self.logger.error(f'Error en el análisis exploratorio de datos: {e}')
            raise

if __name__ == "__main__":
    csv_path = '../data/raw/german_credit_data.csv'
    output_path = '../output'
    dp = DataProcessor(csv_path, output_path)
    dp.cargar_datos()
    dp.limpiar_datos()
    dp.analisis_exploratorio_datos()
```

## Descripción de Métodos

### `__init__`
- Inicializa la clase con la ruta del archivo CSV y la ruta de salida para guardar los resultados del análisis.
- Configura un logger personalizado para registrar las operaciones.

### `cargar_datos`
- Carga los datos desde el archivo CSV especificado.
- Registra un mensaje de éxito si los datos se cargan correctamente, o un mensaje de error si ocurre algún problema.

### `limpiar_datos`
- Llena los valores nulos con la cadena `'unknown'`.
- Convierte las columnas de tipo objeto a tipo categoría para optimizar el espacio y facilitar el análisis.
- Registra un mensaje de éxito si la limpieza se realiza correctamente, o un mensaje de error si ocurre algún problema.

### `analisis_exploratorio_datos`
- Realiza un análisis exploratorio de los datos, generando varias visualizaciones y guardándolas en la ruta de salida.
- Análisis univariado, incluyendo gráficos de distribución para las variables numéricas y categóricas, así como una matriz de correlación.
- Registra un mensaje de éxito si el análisis se completa correctamente, o un mensaje de error si ocurre algún problema.

## Resultados del Análisis Exploratorio de Datos

### Distribución de Riesgo

![Distribución de Riesgo](distribucion_riesgo_1.png)
- **Descripción:** El gráfico muestra la distribución de los valores de riesgo en los datos. A la izquierda se muestra el conteo absoluto de casos con riesgo 'good' y 'bad', mientras que a la derecha se muestra la proporción relativa de estos valores.

### Distribución Numérica

![Distribución Numérica](distribucion_numerica.png)
- **Descripción:** El gráfico muestra la distribución de las variables numéricas (Age, Credit amount, Duration) en los datos. Cada gráfico incluye un histograma y una curva de densidad para visualizar la distribución de los datos.

### Boxplot Numérico

![Boxplot Numérico](boxplot_numerico.png)
- **Descripción:** El gráfico muestra boxplots para las variables numéricas (Age, Credit amount, Duration). Estos gráficos son útiles para identificar valores atípicos y la dispersión de los datos.

### Distribución Categórica

![Distribución Categórica](distribucion_categorica.png)
- **Descripción:** El gráfico muestra la distribución de las variables categóricas en los datos. Cada gráfico de barras indica el conteo de cada categoría en las variables Sex, Job, Housing, Saving accounts, Checking account, Purpose, y Risk.

### Matriz de Correlación

![Matriz de Correlación](matriz_correlacion.png)
- **Descripción:** El gráfico muestra la matriz de correlación de las variables en los datos. Los valores de la matriz varían entre -1 y 1, donde valores cercanos a 1 indican una fuerte correlación positiva y valores cercanos a -1 indican una fuerte correlación negativa.

## Conclusiones

El análisis exploratorio de datos realizado por la clase `DataProcessor` proporciona una visión clara y detallada de las características principales del conjunto de datos, identificando patrones, distribuciones y relaciones entre las variables. Este análisis es crucial para preparar los datos para su posterior modelado y análisis más profundo.
```