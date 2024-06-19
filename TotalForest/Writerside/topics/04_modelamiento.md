# Modelamiento: Ramdom Forest y Decision Tree

## Descripción General
La clase `ModelTraining` es responsable de cargar los datos, entrenar los modelos, evaluar su desempeño y comparar diferentes modelos de aprendizaje automático. Esta clase utiliza modelos de regresión de árbol de decisión (`DecisionTreeRegressor`) y de bosque aleatorio (`RandomForestRegressor`).

## Métodos

### `__init__(self, train_path, test_path, output_path, logger)`
Inicializa la clase con las rutas a los archivos de datos de entrenamiento y prueba, la ruta de salida para guardar los resultados y el logger para registrar eventos.

### `load_data(self)`
Carga los datos de entrenamiento y prueba desde archivos CSV y separa las características (X) de la variable objetivo (y). También codifica la variable objetivo `Risk` utilizando `LabelEncoder`.

```python
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
```

### `train_and_evaluate(self, model, X_train, y_train, X_test, y_test)`
Entrena y evalúa un modelo específico utilizando validación cruzada, ajusta el modelo a los datos de entrenamiento, predice y evalúa el desempeño en los conjuntos de datos de entrenamiento y prueba, y guarda el modelo entrenado.

```python
def train_and_evaluate(self, model, X_train, y_train, X_test, y_test):
    try:
        print(f"Training model: {model.__class__.__name__}")
        cv_results = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        mean_mse = -np.mean(cv_results)
        std_mse = np.std(cv_results)

        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        r2_train = r2_score(y_train, y_train_pred)

        y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        r2_test = r2_score(y_test, y_test_pred)

        print(f"Model: {model.__class__.__name__}, Mean MSE: {mean_mse}, Std MSE: {std_mse}, MSE Train: {mse_train}, MAE Train: {mae_train}, R2 Train: {r2_train}, MSE Test: {mse_test}, MAE Test: {mae_test}, R2 Test: {r2_test}")

        # Guardar el modelo entrenado
        model_path = os.path.join(self.output_path, f'{model.__class__.__name__}.joblib')
        joblib.dump(model, model_path)
        print(f'Modelo {model.__class__.__name__} guardado en {model_path}')
        self.logger.info(f'Modelo {model.__class__.__name__} guardado en {model_path}')

        return mean_mse, std_mse, mse_train, mae_train, r2_train, mse_test, mae_test, r2_test
    except Exception as e:
        self.logger.error(f'Error al entrenar y evaluar el modelo {model.__class__.__name__}: {e}')
        print(f'Error al entrenar y evaluar el modelo {model.__class__.__name__}: {e}')
        raise
```

### `compare_models(self, X_train, y_train, X_test, y_test)`
Compara el desempeño de varios modelos (en este caso, `DecisionTreeRegressor` y `RandomForestRegressor`) utilizando los métodos `train_and_evaluate`. Los resultados de la comparación se guardan en un archivo CSV y se muestran en una tabla.

```python
def compare_models(self, X_train, y_train, X_test, y_test):
    try:
        results = []
        models = [DecisionTreeRegressor(), RandomForestRegressor()]

        for model in models:
            print(f"Evaluando modelo: {model.__class__.__name__}")
            mean_mse, std_mse, mse_train, mae_train, r2_train, mse_test, mae_test, r2_test = self.train_and_evaluate(model, X_train, y_train, X_test, y_test)
            results.append({
                'model': model.__class__.__name__,
                'mean_mse': mean_mse,
                'std_mse': std_mse,
                'mse_train': mse_train,
                'mae_train': mae_train,
                'r2_train': r2_train,
                'mse_test': mse_test,
                'mae_test': mae_test,
                'r2_test': r2_test
            })
            print(f'{model.__class__.__name__}: mean_mse={mean_mse}, std_mse={std_mse}, mse_train={mse_train}, mae_train={mae_train}, r2_train={r2_train}, mse_test={mse_test}, mae_test={mae_test}, r2_test={r2_test}')

        results_df = pd.DataFrame(results)
        results_path = os.path.join(self.output_path, 'model_comparison_results.csv')
        results_df.to_csv(results_path, index=False)
        self.logger.info('Resultados de la comparación de modelos guardados.')
        print('Resultados de la comparación de modelos guardados.')

        # Imprimir resultados en una tabla bonita
        table = PrettyTable()
        table.field_names = ["Model", "Mean MSE", "Std MSE", "MSE Train", "MAE Train", "R2 Train", "MSE Test", "MAE Test", "R2 Test"]
        for result in results:
            table.add_row([
                result['model'],
                f"{result['mean_mse']:.5f}",
                f"{result['std_mse']:.5f}",
                f"{result['mse_train']:.5f}",
                f"{result['mae_train']:.5f}",
                f"{result['r2_train']:.5f}",
                f"{result['mse_test']:.5f}",
                f"{result['mae_test']:.5f}",
                f"{result['r2_test']:.5f}"
            ])
        print(table)
    except Exception as e:
        self.logger.error(f'Error al comparar modelos: {e}')
        print(f'Error al comparar modelos: {e}')
        raise
```

## Resultados de la Comparación de Modelos

### DecisionTreeRegressor

- Mean MSE: 0.32500
- Std MSE: 0.04312
- MSE Train: 0.00000
- MAE Train: 0.00000
- R2 Train: 1.00000
- MSE Test: 0.35500
- MAE Test: 0.35500
- R2 Test: -0.70694

### RandomForestRegressor

- Mean MSE: 0.18325
- Std MSE: 0.00822
- MSE Train: 0.02546
- MAE Train: 0.12689
- R2 Train: 0.87906
- MSE Test: 0.16577
- MAE Test: 0.33320
- R2 Test: 0.20293

## Conclusión
La clase `ModelTraining` permite cargar, entrenar y evaluar modelos de manera efectiva, y proporciona una comparación clara entre diferentes modelos de aprendizaje automático. En este caso, `RandomForestRegressor` mostró un mejor desempeño general en comparación con `DecisionTreeRegressor`.
