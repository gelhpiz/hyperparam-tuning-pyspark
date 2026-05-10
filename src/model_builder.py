import pyspark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator

def build_param_grid(rf):
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 50, 100]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.minInstancesPerNode, [1, 2, 4]) \
        .build()
    return paramGrid

def cross_validate(train, rf, paramGrid, evaluator, numFolds=3):
    crossval = CrossValidator(estimator=rf,
                              estimatorParamMaps=paramGrid,
                              evaluator=evaluator,
                              numFolds=numFolds,
                              parallelism=1)
    cv_model = crossval.fit(train)
    return cv_model

def train_validation_split(train, rf, paramGrid, evaluator, trainRatio=0.75):
    tvs = TrainValidationSplit(estimator=rf,
                               estimatorParamMaps=paramGrid,
                               evaluator=evaluator,
                               trainRatio=trainRatio,
                               parallelism=1)
    tvs_model = tvs.fit(train)
    return tvs_model

def get_best_model_info(model):
    best = model.bestModel
    return {
        "numTrees": best.getNumTrees,
        "maxDepth": best.getMaxDepth(),
        "minInstancesPerNode": best.getMinInstancesPerNode(),
        "avg_metric": model.avgMetrics[0] if hasattr(model, 'avgMetrics') else model.validationMetrics[0]
    }