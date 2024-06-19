import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from src.logger import CustomLogger
from prettytable import PrettyTable

class NeuralNetworkTraining:
    def __init__(self, train_path, test_path, output_path, logger):
        self.train_path = train_path
        self.test_path = test_path
        self.output_path = output_path
        self.logger = logger

    def load_data(self):
        try:
            train_data = pd.read_csv(self.train_path)
            test_data = pd.read_csv(self.test_path)

            print("Datos de entrenamiento cargados:")
            print(train_data.head())

            print("Datos de prueba cargados:")
            print(test_data.head())

            y_train = train_data['Risk']
            X_train = train_data.drop(columns=['Risk'])
            y_test = test_data['Risk']
            X_test = test_data.drop(columns=['Risk'])

            print("X_train:")
            print(X_train.head())

            print("y_train antes de codificar:")
            print(y_train.head())

            # Codificar la variable objetivo
            encoder = LabelEncoder()
            y_train = encoder.fit_transform(y_train)
            y_test = encoder.transform(y_test)

            print("y_train después de codificar:")
            print(y_train[:5])

            self.logger.info('Datos cargados para el modelado.')
            print("Datos cargados para el modelado.")
            return X_train, X_test, y_train, y_test
        except Exception as e:
            self.logger.error(f'Error al cargar los datos: {e}')
            print(f'Error al cargar los datos: {e}')
            raise

    def build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test):
        try:
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

            y_train_pred = model.predict(X_train).flatten()
            y_test_pred = model.predict(X_test).flatten()

            mse_train = mean_squared_error(y_train, y_train_pred)
            mae_train = mean_absolute_error(y_train, y_train_pred)
            r2_train = r2_score(y_train, y_train_pred)

            mse_test = mean_squared_error(y_test, y_test_pred)
            mae_test = mean_absolute_error(y_test, y_test_pred)
            r2_test = r2_score(y_test, y_test_pred)

            print(f"Model: NeuralNetwork, MSE Train: {mse_train}, MAE Train: {mae_train}, R2 Train: {r2_train}")
            print(f"Model: NeuralNetwork, MSE Test: {mse_test}, MAE Test: {mae_test}, R2 Test: {r2_test}")

            # Guardar el modelo
            model_path = os.path.join(self.output_path, 'neural_network_model.h5')
            model.save(model_path)
            self.logger.info(f'Modelo NeuralNetwork guardado en {model_path}')
            print(f'Modelo NeuralNetwork guardado en {model_path}')

            return mse_train, mae_train, r2_train, mse_test, mae_test, r2_test
        except Exception as e:
            self.logger.error(f'Error al entrenar y evaluar el modelo NeuralNetwork: {e}')
            print(f'Error al entrenar y evaluar el modelo NeuralNetwork: {e}')
            raise

    def compare_models(self, X_train, y_train, X_test, y_test):
        try:
            results = []

            # Cargar resultados anteriores si existen
            results_path = os.path.join(self.output_path, 'model_comparison_results.csv')
            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                results = results_df.to_dict('records')

            # Remover entradas previas de NeuralNetwork
            results = [result for result in results if result['model'] != 'NeuralNetwork']

            # Construir y entrenar el modelo de red neuronal
            input_dim = X_train.shape[1]
            model = self.build_model(input_dim)
            mse_train, mae_train, r2_train, mse_test, mae_test, r2_test = self.train_and_evaluate(model, X_train, y_train, X_test, y_test)
            results.append({
                'model': 'NeuralNetwork',
                'mean_mse': 'N/A',  # No calculamos mean_mse y std_mse para la red neuronal
                'std_mse': 'N/A',
                'mse_train': mse_train,
                'mae_train': mae_train,
                'r2_train': r2_train,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'r2_test': r2_test
            })

            results_df = pd.DataFrame(results)
            results_df.to_csv(results_path, index=False)
            self.logger.info('Resultados de la comparación de modelos guardados.')
            print('Resultados de la comparación de modelos guardados.')

            # Imprimir resultados en una tabla bonita
            table = PrettyTable()
            table.field_names = ["Model", "Mean MSE", "Std MSE", "MSE Train", "MAE Train", "R2 Train", "MSE Test", "MAE Test", "R2 Test"]
            for result in results:
                table.add_row([
                    result['model'],
                    f"{result['mean_mse']}" if 'mean_mse' in result else 'N/A',
                    f"{result['std_mse']}" if 'std_mse' in result else 'N/A',
                    f"{result['mse_train']:.5f}" if 'mse_train' in result else 'N/A',
                    f"{result['mae_train']:.5f}" if 'mae_train' in result else 'N/A',
                    f"{result['r2_train']:.5f}" if 'r2_train' in result else 'N/A',
                    f"{result['mse_test']:.5f}" if 'mse_test' in result else 'N/A',
                    f"{result['mae_test']:.5f}" if 'mae_test' in result else 'N/A',
                    f"{result['r2_test']:.5f}" if 'r2_test' in result else 'N/A'
                ])
            print(table)
        except Exception as e:
            self.logger.error(f'Error al comparar modelos: {e}')
            print(f'Error al comparar modelos: {e}')
            raise


# Uso de la clase NeuralNetworkTraining
if __name__ == "__main__":
    TRAIN_PATH = '../output/train_data.csv'
    TEST_PATH = '../output/test_data.csv'
    OUTPUT_PATH = '../output'

    # Instanciar el logger
    logger = CustomLogger().get_logger()

    # Crear instancia de NeuralNetworkTraining
    nn_trainer = NeuralNetworkTraining(TRAIN_PATH, TEST_PATH, OUTPUT_PATH, logger)

    # Cargar los datos
    X_train, X_test, y_train, y_test = nn_trainer.load_data()

    # Comparar modelos
    nn_trainer.compare_models(X_train, y_train, X_test, y_test)
