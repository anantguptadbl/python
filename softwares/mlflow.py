# The following is a very simple example on how to use mlflow

# Installation Instructions
# pip install mlflow

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

warnings.filterwarnings("ignore")
np.random.seed(40)

# Random Data
data=pd.DataFrame(np.random.rand(1000,10),columns=['col{0}'.format(x) for x in range(10)])
# Predicted Column
data['target']=np.random.randint(10,size=(1000))
# Feature Column Names
targetColumn='target'
featureColumns=['col{0}'.format(x) for x in range(10)]

# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)

# Model Parameters on which we will do grid search, and we will make use of mlflow to log the details for us
for alpha in [0.2,0.5,0.7,0.9]:
    for l1_ratio in [0.2,0.5,0.7,0.9]:
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train[featureColumns], train[targetColumn])

            predictions = lr.predict(test[featureColumns])

            (rmse, mae, r2) = eval_metrics(test[targetColumn], predictions)
            
            # We need to mention the metrics that we want to log
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)
            mlflow.log_param("model","elasticNetModel")
            mlflow.sklearn.log_model(lr, "elasticNetModel")
            
print("Cell Execution Completed")


# Now on the command line, if we write 
# prompt> mlflow ui
# Serving on http://127.0.0.1:5000

