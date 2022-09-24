import pandas as pd
from db_connections import DatabaseOperations

'''
from Database_Operations.db_connections import DatabaseOperations
from Pre_Processing.pre_process import Process
from Pre_Processing.feature_engineering import Engineering
from Clustering_Technique.cluster_dataset import ClusteringDataset
from Clustering_Technique.divide_datasets_for_clusters import DivideDatasets
from Clustering_Technique.cluster_models import MakeClusterModels
from Basic_Models_Implementation.basic_models import MakeBasicModels
from Predictions.pred import MakePredictions
'''
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
# from EDA.eda import Preprocess
from sklearn.metrics import plot_confusion_matrix


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Inputs to be taken
# Database or csv upload
# Label Column
# Automate model_type and manual both
# Clustering - "True" or "False", as string (currently)
# Models to prepare for classification and regression
# Automate model for training part

regression_models = ["Random Forest Regressor", "SVM Regressor", "Linear Regressor", "XGBoost Regressor"]
clf_models = ["Random Forest Classifier", "SVM Classifier", "NaiveBayes Classifier", "XGBoost Classifier"]
df = pd.DataFrame()

db_type = input("Select the database type(Enter the number): \n 1.Mongo \n 2.Upload \n")

if db_type == "Mongo":

    print("Type the details of the database.\n")
    port = input("Port of Database: ")
    host_address = input("Host Address: ")
    username = input("Username: ")
    password = input("Password: ")
    database_name = input("Database Name: ")
    collection_name = input("Collection Name: ")

    # Verification with the database
    if port and host_address and database_name and collection_name:
        df = DatabaseOperations().make_csv_from_mongo(port=port, host_address=host_address, database_name=database_name,
                                                      collection_name=collection_name)

    if port and host_address and username and password and database_name and collection_name:
        df = DatabaseOperations().make_csv_from_mongo(port=port, host_address=host_address, username=username,
                                                      password=password, database_name=database_name,
                                                      collection_name=collection_name)

elif db_type == "Upload":
    file_name = input("Enter name of your csv file: ")
    df = pd.read_csv(file_name)

process_type = input("Do you want to automate the process? \n Enter 'Automate' for automation or "
                 "'Manual' to manually proceed. \n")

# Automate (Modelling)
if not df.empty and automate == "Automate":

    print("Connection established! ")
    columns_list = list(df.columns)
    print("From the following columns select the dependent feature: \n", columns_list)
    dep_feat = input("Enter the dependent feature: ")

    if df[dep_feat].value_counts().count() < 3:
        print("Classification model will be used. ")
        print("Following models will be used: \n", clf_models)
        model_list = clf_models

    else:
        print("Regression model will be used. ")
        print("Following models will be used: \n", regression_models)
        model_list = regression_models

# Manual
if not df.empty and process_type == "Manual":
    print("Connection established! ")
    page = input("Select the task: \n 1. Modelling \n 2. EDA \n")
    if page == "Modelling":
        columns_list = list(df.columns)
        model_type = input("Select the problem type: \n 1. Regression \n 2. Classification \n")
        print("From the column list choose the label column: \n ", columns_list)
        label_column = input("Enter the label column: \n")
        cluster_bool = input("Do clustering? \n 1. True \n 2. False \n")
        # regression_models = ["Random Forest Regressor", "SVM Regressor", "Linear Regressor", "XGBoost Regressor"]
        # clf_models = ["Random Forest Classifier", "SVM Classifier", "NaiveBayes Classifier", "XGBoost Classifier"]
        model_count = int(input("Enter the number of models you want to use: "))
        model_list = list()
        if model_type == "Regression":
            print("Following regressors are available: \n", regression_models)
            for i in range(model_count):
                select_model = input("Enter model {i}: ".format(i = i+1))
                model_list.append(select_model)

        elif model_type == "Classification":
            print("Following classifiers are available: \n", clf_models)
            for i in range(model_count):
                select_model = input("Enter model {i}: ".format(i = i+1))
                model_list.append(select_model)

        print("The following models were selected: \n", model_list)

    elif page == "EDA":
        columns_list = df.columns
        print("From the column list select the dependent feature: \n", columns_list)
        dep_feat = input("Enter the dependent feature: ")
