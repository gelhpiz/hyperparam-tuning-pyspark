# Ajuste de Hiperparámetros con PySpark

Este proyecto implementa la búsqueda de hiperparámetros para un modelo de Random Forest Regressor sobre el dataset **California Housing**, utilizando PySpark MLlib.

## Características

- Compara dos estrategias de búsqueda: **CrossValidator** (3 folds) y **TrainValidationSplit** (75% entrenamiento, 25% validación).
- Evalúa el rendimiento de los modelos optimizados frente al modelo por defecto.
- Visualiza la importancia de las características.
- Guarda el mejor modelo para su uso futuro.

## Dataset

El dataset se descarga automáticamente desde:
[https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv](https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv)

## Requisitos

- Python 3.10+
- PySpark 3.4+
- Pandas, Matplotlib, Seaborn

## Instalación y ejecución

## Instalar Java 
conda install -c conda-forge openjdk=11

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/hyperparam-tuning-pyspark.git
cd hyperparam-tuning-pyspark

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar pipeline
python run.py

