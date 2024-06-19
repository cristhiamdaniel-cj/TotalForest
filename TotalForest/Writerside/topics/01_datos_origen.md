# German Credit Risk - With Target

## Descripción del Conjunto de Datos

Este conjunto de datos contiene información sobre los créditos otorgados en Alemania, junto con la clasificación de riesgo correspondiente. El conjunto de datos es útil para entrenar modelos de machine learning para la predicción del riesgo crediticio.

### Origen de los Datos

El conjunto de datos se puede encontrar en Kaggle: [German Credit Data - With Risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk/data).

### Descripción de las Columnas

El conjunto de datos contiene las siguientes columnas:

- **Unnamed: 0**: Identificador del registro.
- **Age**: La edad del cliente.
- **Credit amount**: La cantidad de crédito otorgado.
- **Duration**: La duración del crédito en meses.
- **Sex_male**: Indica si el cliente es hombre (True) o no (False).
- **Job_1, Job_2, Job_3**: Variables dummy que indican el tipo de trabajo del cliente.
- **Housing_own, Housing_rent**: Variables dummy que indican el tipo de vivienda del cliente.
- **Saving accounts_moderate, Saving accounts_quite rich, Saving accounts_rich, Saving accounts_unknown**: Variables dummy que indican el tipo de cuenta de ahorros del cliente.
- **Checking account_moderate, Checking account_rich, Checking account_unknown**: Variables dummy que indican el tipo de cuenta corriente del cliente.
- **Purpose_car, Purpose_domestic appliances, Purpose_education, Purpose_furniture/equipment, Purpose_radio/TV, Purpose_repairs, Purpose_vacation/others**: Variables dummy que indican el propósito del crédito.
- **Risk_good**: Indica si el riesgo del crédito es bueno (True) o no (False).

### Muestra de los Datos

| Unnamed: 0 | Age | Credit amount | Duration | Sex_male | Job_1 | Job_2 | Job_3 | Housing_own | Housing_rent | Saving accounts_moderate | Saving accounts_quite rich | Saving accounts_rich | Saving accounts_unknown | Checking account_moderate | Checking account_rich | Checking account_unknown | Purpose_car | Purpose_domestic appliances | Purpose_education | Purpose_furniture/equipment | Purpose_radio/TV | Purpose_repairs | Purpose_vacation/others | Risk_good |
|------------|-----|---------------|----------|----------|-------|-------|-------|-------------|--------------|--------------------------|----------------------------|----------------------|-------------------------|--------------------------|----------------------|-------------------------|--------------|----------------------------|------------------|-----------------------------|-----------------|-----------------|-------------------------|-----------|
| 0          | 67  | 1169          | 6        | True     | False | True  | False | True        | False        | False                    | False                      | False                | True                    | False                    | False                | False                  | False        | False                      | False            | False                       | True            | False          | False                   | True      |
| 1          | 22  | 5951          | 48       | False    | False | True  | False | True        | False        | False                    | False                      | False                | False                   | True                     | False                | False                  | False        | False                      | False            | False                       | True            | False          | False                   | False     |
| 2          | 49  | 2096          | 12       | True     | True  | False | False | True        | False        | False                    | False                      | False                | False                   | False                    | False                | True                   | False        | False                      | True             | False                       | False           | False          | False                   | True      |
| 3          | 45  | 7882          | 42       | True     | False | True  | False | False       | False        | False                    | False                      | False                | False                   | False                    | False                | False                  | False        | False                      | False            | True                        | False           | False          | False                   | True      |
| 4          | 53  | 4870          | 24       | True     | False | True  | False | False       | False        | False                    | False                      | False                | False                   | False                    | False                | False                  | True         | False                      | False            | False                       | False           | False          | False                   | False     |

### Información de las Columnas

- **Conteo de Registros**: 1000
- **Columnas**: 25
- **Tipos de Datos**:
    - **bool**: 21
    - **int64**: 4
- **Uso de Memoria**: 51.9 KB

### Estadísticas Descriptivas

|                  | Unnamed: 0   | Age         | Credit amount | Duration    |
|------------------|--------------|-------------|---------------|-------------|
| **count**        | 1000.000000  | 1000.000000 | 1000.000000   | 1000.000000 |
| **mean**         | 499.500000   | 35.546000   | 3271.258000   | 20.903000   |
| **std**          | 288.819436   | 11.375469   | 2822.736876   | 12.058814   |
| **min**          | 0.000000     | 19.000000   | 250.000000    | 4.000000    |
| **25%**          | 249.750000   | 27.000000   | 1365.500000   | 12.000000   |
| **50%**          | 499.500000   | 33.000000   | 2319.500000   | 18.000000   |
| **75%**          | 749.250000   | 42.000000   | 3972.250000   | 24.000000   |
| **max**          | 999.000000   | 75.000000   | 18424.000000  | 72.000000   |

### Conclusión

El conjunto de datos de crédito alemán es una herramienta valiosa para desarrollar y evaluar modelos de predicción de riesgo crediticio. La variedad de características y la presencia de una variable objetivo bien definida permiten realizar análisis profundos y construir modelos de machine learning efectivos.
