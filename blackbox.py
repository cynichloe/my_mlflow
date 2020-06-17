# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import random

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def read_in_data():
    csv_url =\
        'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
    try:
        return pd.read_csv(csv_url, sep=';')
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)

def seperate_data(test: pd.DataFrame, train: pd.DataFrame, label_column: str):
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    return train_x, test_x, train_y, test_y


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # np.random.seed(40)

    # Read the wine-quality csv file from the URL
    data = read_in_data()
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)
    train_x, test_x, train_y, test_y = seperate_data(train=train, test=test, label_column='quality')

    level_a = int(sys.argv[1])
    level_b = int(sys.argv[2])
    no_of_runs = int(sys.argv[3])
    base = pow(10, int(level_b/10))

    # print(type(level_a), type(level_b), type(no_of_runs))
    # print(level_a, level_b, no_of_runs)

    alpha_list = [random.randint(level_a, level_b)/base for i in range(0,no_of_runs)]
    l1_ratio_list = [random.randint(level_a, level_b)/base for i in range(0,no_of_runs)]
    params_list = zip(alpha_list, l1_ratio_list)

    mlflow.set_experiment(f"exp_fixed_l1")

    for param_pair in params_list:
        alpha = param_pair[0]
        l1_ratio = param_pair[1]

        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            predicted_qualities = lr.predict(test_x)

            (rmse, mae, r2) = eval_metrics(actual=test_y, pred=predicted_qualities)

            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
            print(f"RMSE: {rmse} , MAE: {mae}, R2: {r2}")

            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            if tracking_url_type_store != "file":
                # Register the model
                # There are other ways to use the Model Registry, which depends on the use case,
                # please refer to the doc for more information:
                # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
            else:
                mlflow.sklearn.log_model(lr, "model")
