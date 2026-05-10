import requests
import os
from pyspark.sql import SparkSession

def download_housing_data(url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv",
                          dest="data/housing.csv"):
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(dest):
        print(f"Descargando {url} -> {dest}")
        r = requests.get(url)
        with open(dest, "wb") as f:
            f.write(r.content)
    else:
        print(f"Archivo ya existe: {dest}")
    return dest

def load_and_prepare_data(spark, filepath):
    df = spark.read.option("inferSchema", "true").option("header", "true").csv(filepath)
    feature_cols = [c for c in df.columns if c not in ['median_house_value', 'ocean_proximity']]
    
    # ELIMINAR FILAS CON VALORES NULOS
    df_clean = df.na.drop(subset=feature_cols + ['median_house_value'])
    
    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
    data = assembler.transform(df_clean).select("features", "median_house_value")
    
    train, test = data.randomSplit([0.8, 0.2], seed=42)
    return train, test, feature_cols