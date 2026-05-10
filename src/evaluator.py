import pyspark
from pyspark.ml.evaluation import RegressionEvaluator

def evaluate_model(model, test, metric_name="rmse"):
    evaluator = RegressionEvaluator(labelCol="median_house_value", metricName=metric_name)
    predictions = model.transform(test)
    metric = evaluator.evaluate(predictions)
    return metric, predictions

def compare_models(default_model, cv_model, tvs_model, test):
    evaluator = RegressionEvaluator(labelCol="median_house_value", metricName="rmse")
    default_metric = evaluator.evaluate(default_model.transform(test))
    cv_metric = evaluator.evaluate(cv_model.transform(test))
    tvs_metric = evaluator.evaluate(tvs_model.transform(test))
    return {
        "default": default_metric,
        "cross_validator": cv_metric,
        "train_validation_split": tvs_metric
    }