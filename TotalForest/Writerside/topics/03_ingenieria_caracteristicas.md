# Feature Engineering

## Descripción General

La clase `FeatureEngineering` se encarga de la creación, escalado y división de características del conjunto de datos de crédito. Su objetivo es preparar los datos para su posterior uso en modelos de machine learning.

## Código

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import os
from src.logger import CustomLogger

class FeatureEngineering:
    def __init__(self, data, output_path, logger):
        self.data = data
        self.output_path = output_path
        self.logger = logger

    def create_features(self):
        try:
            # Ejemplo de creación de una nueva característica
            self.data['CreditAmountPerMonth'] = self.data['Credit amount'] / self.data['Duration']
            self.logger.info('Nuevas características creadas.')
            print("Nuevas características creadas.")
            return self.data
        except Exception as e:
            self.logger.error(f'Error al crear nuevas características: {e}')
            print(f'Error al crear nuevas características: {e}')
            raise

    def scale_features(self):
        try:
            # Separar la variable objetivo antes de las transformaciones
            y = self.data['Risk']
            X = self.data.drop(columns=['Risk'])

            # Imprimir los datos después de separar la variable objetivo
            print("Datos después de separar la variable objetivo:")
            print(X.head())

            # Separar características numéricas y categóricas
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
            categorical_features = X.select_dtypes(include=['object', 'category']).columns

            # Imprimir las características numéricas y categóricas
            print("Características numéricas:", numeric_features)
            print("Características categóricas:", categorical_features)

            # Pipelines de preprocesamiento para características numéricas y categóricas
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            # Aplicar preprocesamiento
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            # Ajustar el preprocesador y transformar los datos
            print("Ajustando el preprocesador...")
            X_processed = preprocessor.fit_transform(X)
            print("Preprocesador ajustado y datos transformados.")

            # Obtener los nombres de las características resultantes
            feature_names_num = numeric_features.tolist()
            print("Nombres de características numéricas:", feature_names_num)

            # Comprobar si hay características categóricas antes de intentar obtener sus nombres
            if categorical_features.size > 0:
                feature_names_cat = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features).tolist()
                print("Nombres de características categóricas:", feature_names_cat)
                feature_names = feature_names_num + feature_names_cat
            else:
                feature_names = feature_names_num

            print("Nombres de todas las características:", feature_names)

            # Convertir el resultado de vuelta a un DataFrame
            X_processed = pd.DataFrame(X_processed, columns=feature_names)
            X_processed['Risk'] = y.reset_index(drop=True)

            self.data = X_processed

            # Guardar los datos escalados
            scaled_data_path = os.path.join(self.output_path, 'scaled_data.csv')
            self.data.to_csv(scaled_data_path, index=False)

            self.logger.info('Características escaladas y datos guardados.')
            print("Características escaladas y datos guardados.")
            return self.data
        except Exception as e:
            self.logger.error(f'Error al escalar características: {e}')
            print(f'Error al escalar características: {e}')
            raise

    def split_data(self, test_size=0.2, random_state=42):
        try:
            # Separar la variable objetivo y las características
            y = self.data['Risk']
            X = self.data.drop(columns=['Risk'])

            # Dividir los datos en conjuntos de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

            # Guardar los conjuntos de datos en archivos
            train_data = pd.DataFrame(X_train)
            test_data = pd.DataFrame(X_test)
            train_data['Risk'] = y_train
            test_data['Risk'] = y_test

            train_path = os.path.join(self.output_path, 'train_data.csv')
            test_path = os.path.join(self.output_path, 'test_data.csv')

            train_data.to_csv(train_path, index=False)
            test_data.to_csv(test_path, index=False)

            self.logger.info('Datos divididos en conjuntos de entrenamiento y prueba.')
            print("Datos divididos en conjuntos de entrenamiento y prueba.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f'Error al dividir los datos: {e}')
            print(f'Error al dividir los datos: {e}')
            raise


# Uso de la clase FeatureEngineering
if __name__ == "__main__":
    CSV_PATH = '../data/raw/german_credit_data.csv'
    OUTPUT_PATH = '../output'

    # Instanciar el logger
    logger = CustomLogger().get_logger()

    # Cargar datos
    data = pd.read_csv(CSV_PATH)

    # Crear instancia de FeatureEngineering
    feature_engineer = FeatureEngineering(data, OUTPUT_PATH, logger)

    # Crear nuevas características
    feature_engineer.create_features()

    # Escalar características
    feature_engineer.scale_features()

    # Dividir el conjunto de datos
    feature_engineer.split_data()
```

## Descripción de Métodos

### `__init__`
- Inicializa la clase con los datos, la ruta de salida y el logger.

### `create_features`
- Crea una nueva característica `CreditAmountPerMonth` dividiendo el monto del crédito por la duración del crédito.
- Registra un mensaje de éxito si la creación de características se realiza correctamente, o un mensaje de error si ocurre algún problema.

### `scale_features`
- Separa la variable objetivo (`Risk`) de las características.
- Define y aplica pipelines de preprocesamiento para características numéricas y categóricas utilizando `StandardScaler` y `OneHotEncoder`.
- Transforma los datos y guarda el conjunto de datos escalados en un archivo CSV.
- Registra un mensaje de éxito si el escalado de características se realiza correctamente, o un mensaje de error si ocurre algún problema.

### `split_data`
- Separa la variable objetivo (`Risk`) de las características.
- Divide los datos en conjuntos de entrenamiento y prueba utilizando `train_test_split`.
- Guarda los conjuntos de datos en archivos CSV.
- Registra un mensaje de éxito si la división de datos se realiza correctamente, o un mensaje de error si ocurre algún problema.

## Resultados

### Creación de Nuevas Características

```plaintext
Nuevas características creadas.
```

- **Descripción:** La nueva característica `CreditAmountPerMonth` se crea dividiendo el monto del crédito por la duración del crédito.

### Escalado de Características

```plaintext
Datos después de separar la variable objetivo:
   Unnamed: 0  Age  Credit amount  Duration  ...  Saving accounts  Checking account  Purpose  CreditAmountPerMonth
0          67   30          1000        24  ...              0.0               0.0      0.0              41.666667
1          22   22          5951        48  ...              0.0               0.0      0.0             124.020833
2          49   49          2096        12  ...              0.0               0.0      0.0             174.666667
3          45   45          7882        42  ...              0.0               0.0      0.0             187.666667
4          53   53          4870        24  ...              0.0               0.0      0.0             202.916667

Características numéricas: Index(['Unnamed: 0', 'Age', 'Credit amount', 'Duration', 'CreditAmountPerMonth'], dtype='object')
Características categóricas: Index(['Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose'], dtype='object')

Ajustando el preprocesador...
Preprocesador ajustado y datos transformados.

Nombres de características numéricas: ['Unnamed: 0', 'Age', 'Credit amount', 'Duration', 'CreditAmountPerMonth']
Nombres de características categóricas: ['Sex_female', 'Sex_male', 'Job_0', 'Job_1', 'Job_2', 'Job_3', 'Housing_free', 'Housing_own', 'Housing_rent', 'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_quite rich', 'Saving accounts_rich', 'Saving accounts_unknown', 'Checking account_little', 'Checking account_moderate', 'Checking account_rich', 'Checking account_unknown', 'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', '```markdown
'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others']
Nombres de todas las características: ['Unnamed: 0', 'Age', 'Credit amount', 'Duration', 'CreditAmountPerMonth', 'Sex_female', 'Sex_male', 'Job_0', 'Job_1', 'Job_2', 'Job_3', 'Housing_free', 'Housing_own', 'Housing_rent', 'Saving accounts_little', 'Saving accounts_moderate', 'Saving accounts_quite rich', 'Saving accounts_rich', 'Saving accounts_unknown', 'Checking account_little', 'Checking account_moderate', 'Checking account_rich', 'Checking account_unknown', 'Purpose_business', 'Purpose_car', 'Purpose_domestic appliances', 'Purpose_education', 'Purpose_furniture/equipment', 'Purpose_radio/TV', 'Purpose_repairs', 'Purpose_vacation/others']

Características escaladas y datos guardados.
```

- **Descripción:** Los datos se escalan utilizando `StandardScaler` para características numéricas y `OneHotEncoder` para características categóricas. Los datos escalados se guardan en `scaled_data.csv`.

### División de Datos

```plaintext
Datos divididos en conjuntos de entrenamiento y prueba.
```

- **Descripción:** Los datos se dividen en conjuntos de entrenamiento y prueba con un tamaño de prueba del 20%. Los conjuntos de datos resultantes se guardan en `train_data.csv` y `test_data.csv`.

## Conclusión

La clase `FeatureEngineering` facilita la preparación de datos mediante la creación de nuevas características, el escalado de datos y la división en conjuntos de entrenamiento y prueba. Esto es esencial para el modelado efectivo y preciso en machine learning.
