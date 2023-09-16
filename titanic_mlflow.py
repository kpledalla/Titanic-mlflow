import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold,  GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score, mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("Titanic+Data+Set.csv")
dff= df.copy()

# replace NaN in cabin column with its mode
dff['Cabin'].fillna(dff['Cabin'].mode()[0], inplace=True)

#replace NaN in age with median
dff['Age'].fillna(dff['Age'].median(), inplace=True)

dff.drop(columns=["PassengerId", "SibSp", "Name"], inplace=True)

features = ["Embarked", "Cabin", "Ticket", "Sex"]
for col in features:
     dff[col] = LabelEncoder().fit_transform(dff[col])

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

mlflow.set_experiment(experiment_name = 'titanic')
with mlflow.start_run():
    trainsize=0.7
    testsize=0.3
    estimators=100
    maxdepth=10
    x = dff.drop(["Survived"], axis=1)
    Y = dff[["Survived"]]
    x_train, x_test, y_train, y_test = train_test_split(x, Y, train_size=trainsize, test_size=testsize, random_state=22)

    rf_clf = RandomForestClassifier(n_estimators=estimators, random_state=11, max_depth=maxdepth)
    # rf_clf = DecisionTreeClassifier(criterion = 'entropy' )
    rf_clf.fit(x_train, y_train)
        

    y_pred = rf_clf.predict(x_test)
        
        # calculate accuracy for test set
    test_acc = accuracy_score(y_test, y_pred)
    (rmse, mae, r2) = eval_metrics(y_test, y_pred)
        
        # return the test accuracy 
    print('Test Accuracy:', test_acc)
    log_param("n-estimators", rf_clf.n_estimators)
    log_param("max-Depth", rf_clf.max_depth)
    log_param("train-size", trainsize)

    log_metric("test Accuracy", test_acc)
    log_metric("rmse", rmse)
    log_metric("r2", r2)
    log_metric("mae", mae)
    log_metric("f1_score", f1_score(y_test, y_pred))
    log_metric("recall", recall_score(y_test, y_pred))
    log_metric("precision_score", precision_score(y_test, y_pred))
    mlflow.sklearn.log_model(rf_clf, "Model")