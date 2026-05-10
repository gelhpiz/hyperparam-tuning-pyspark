import pyspark
##import findspark
##findspark.init()
from pyspark.sql import SparkSession
from src.data_loader import download_housing_data, load_and_prepare_data
from src.model_builder import (build_param_grid, cross_validate,
                               train_validation_split, get_best_model_info)
from src.evaluator import evaluate_model, compare_models
from src.visualizer import plot_comparison, plot_feature_importance
from pyspark.ml.regression import RandomForestRegressor
import time

def main():
    spark = SparkSession.builder.appName("HyperparamLab").getOrCreate()
    
    # Descargar y cargar datos
    filepath = download_housing_data()
    train, test, feature_cols = load_and_prepare_data(spark, filepath)
    print(f"Entrenamiento: {train.count()}, Prueba: {test.count()}")
    
    # Modelo base
    rf = RandomForestRegressor(featuresCol="features", labelCol="median_house_value", seed=42)
    
    # Grid de parámetros
    paramGrid = build_param_grid(rf)
    print(f"Número total de combinaciones: {len(paramGrid)}")
    
    # Evaluador
    from pyspark.ml.evaluation import RegressionEvaluator
    evaluator = RegressionEvaluator(labelCol="median_house_value", metricName="rmse")
    
    # CrossValidator
    print("Iniciando CrossValidator...")
    start = time.time()
    cv_model = cross_validate(train, rf, paramGrid, evaluator, numFolds=3)
    print(f"CrossValidator completado en {time.time()-start:.2f} segundos")
    cv_info = get_best_model_info(cv_model)
    print("Mejores parámetros (CrossValidator):")
    print(f" - numTrees: {cv_info['numTrees']}")
    print(f" - maxDepth: {cv_info['maxDepth']}")
    print(f" - minInstancesPerNode: {cv_info['minInstancesPerNode']}")
    print(f"Mejor RMSE promedio en validación: {cv_info['avg_metric']:.4f}")
    
    # TrainValidationSplit
    print("Iniciando TrainValidationSplit...")
    start = time.time()
    tvs_model = train_validation_split(train, rf, paramGrid, evaluator, trainRatio=0.75)
    print(f"TrainValidationSplit completado en {time.time()-start:.2f} segundos")
    tvs_info = get_best_model_info(tvs_model)
    print("Mejores parámetros (TrainValidationSplit):")
    print(f" - numTrees: {tvs_info['numTrees']}")
    print(f" - maxDepth: {tvs_info['maxDepth']}")
    print(f" - minInstancesPerNode: {tvs_info['minInstancesPerNode']}")
    print(f"Mejor métrica (RMSE) en validación: {tvs_info['avg_metric']:.4f}")
    
    # Modelo por defecto
    default_model = rf.fit(train)
    
    # Comparación en test
    metrics = compare_models(default_model, cv_model.bestModel, tvs_model.bestModel, test)
    print("Resultados sobre TEST:")
    print(f"Random Forest por defecto     - RMSE: {metrics['default']:.4f}")
    print(f"Random Forest optimizado (CV) - RMSE: {metrics['cross_validator']:.4f}")
    print(f"Random Forest optimizado (TVS)- RMSE: {metrics['train_validation_split']:.4f}")
    
    # Visualización
    plot_comparison(metrics)
    
    # Importancia de características
    best_model = cv_model.bestModel
    plot_feature_importance(best_model, feature_cols)
    
    # Guardar modelo
    model_path = "output/best_rf_model"
    best_model.write().overwrite().save(model_path)
    print(f"Modelo guardado en {model_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()