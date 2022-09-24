import pandas as pd
from flask import Flask, request
from db_connections import DatabaseOperations
from pre_process import Process
from feature_engineering import Engineering
from cluster_dataset import ClusteringDataset
from divide_datasets_for_clusters import DivideDatasets
from cluster_models import MakeClusterModels
from basic_models import MakeBasicModels
from pred import MakePredictions
import pandas as pd

app=Flask(__name__)
@app.route('/',methods=['GET','POST'])
def build_connection():
    port=request.values.get("port")
    host_address=request.values.get("host_address")
    username=request.values.get("username")
    password=request.values.get("password")
    database_name=request.values.get("database_name")
    collection_name=request.values.get("collection_name")
    df = DatabaseOperations().make_csv_from_mongo(port=port, host_address=host_address, username=username, 
                                                   password=password, database_name=database_name, collection_name=collection_name)

    return df.to_json()

@app.route('/details',methods=['GET','POST'])
def get_details():
    df = pd.read_json(build_connection())
    model_type = request.values.get("model_type")
    label_column = request.values.get("label_column")
    clustering_bool = request.values.get("clustering_bool")
    
    model1 = request.values.get("model1")
    model2 = request.values.get("model2")
    model3 = request.values.get("model3")
    list_of_models = [model1,model2,model3]
    
    if clustering_bool == "True":

        preprocessed_dataset, transform_dict = Process(dataset=df, model_type=model_type, label_column=label_column, 
                                                    clustering_bool=clustering_bool).data_preprocessing()

        clustering, number_of_clusters = ClusteringDataset(dataset=preprocessed_dataset, label_column=label_column).make_clusters()

        if model_type == "Regression":
            divided_dataset, standard_scaler, min_max_scaler = DivideDatasets(dataset=preprocessed_dataset, number_of_clusters=number_of_clusters, 
                                                                            label_column=label_column, model_type=model_type, clustering=clustering).divide()
        else:
            divided_dataset, standard_scaler = DivideDatasets(dataset=preprocessed_dataset, number_of_clusters=number_of_clusters, 
                                                                            label_column=label_column, model_type=model_type)

        models = MakeClusterModels(datasets=divided_dataset, model_type=model_type, min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)

        upload_df = pd.read_csv("Datasets/Ride Test Data.csv")
        predicted_df = MakePredictions(data_for_prediction=upload_df, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                min_max_scaler=min_max_scaler, models=models, model_type=model_type, clustering=clustering).predictions()
        
    else:
        
        preprocessed_dataset, transform_dict, standard_scaler, min_max_scaler = Process(dataset=df, model_type=model_type, label_column=label_column, 
                                                                                        clustering_bool=clustering_bool).data_preprocessing()

        models = MakeBasicModels(dataset=preprocessed_dataset, model_type=model_type, label_column=label_column, 
                                min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)
                                
        upload_df = pd.read_csv("Datasets/Ride Test Data.csv")
        predicted_df = MakePredictions(data_for_prediction=upload_df, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                min_max_scaler=min_max_scaler, models=models, model_type=model_type).predictions()

    return predicted_df["Predictions"].to_json()
if __name__=="__main__":
    app.run(debug=True)