# src/feature_engineering.py

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
