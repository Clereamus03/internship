import streamlit as st
import pandas as pd
from db_connections import DatabaseOperations
from pre_process import Process
from feature_engineering import Engineering
from cluster_dataset import ClusteringDataset
from divide_datasets_for_clusters import DivideDatasets
from cluster_models import MakeClusterModels
from basic_models import MakeBasicModels
from pred import MakePredictions
import pandas as pd
import pickle

database_type =st.sidebar.selectbox("Select the Database", ["Select One", "MongoDB", "Redis", "Postgres", "MySQL", "Upload File"])

df = pd.DataFrame()
placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
placeholder4 = st.empty()
placeholder5 = st.empty()
placeholder6 = st.empty()
placeholder_for_info = st.sidebar.empty()

if database_type == "MongoDB":

    placeholder_for_info.info("Type the details of the database.")

    port = placeholder1.text_input('Port of Database')
    host_address = placeholder2.text_input("Host Address")
    username = placeholder3.text_input("Username")
    password = placeholder4.text_input("Password")
    database_name = placeholder5.text_input("Database Name")
    collection_name = placeholder6.text_input("Collection Name")
                           
    if port and host_address and database_name and collection_name:

        df = DatabaseOperations().make_csv_from_mongo(port=port, host_address=host_address, database_name=database_name, 
                                                      collection_name=collection_name)
    
    if port and host_address and username and password and database_name and collection_name:

        df = DatabaseOperations().make_csv_from_mongo(port=port, host_address=host_address, username=username, 
                                                    password=password, database_name=database_name, collection_name=collection_name)
elif database_type == "Upload File":
    placeholder_for_info.info("Upload the file.")
    dataset = placeholder1.file_uploader("Upload your CSV file", type=["csv"], key="dataset_uploader")
    if dataset:
        df =pd.read_csv(dataset)

if df.empty == False:
    
    placeholder_for_info.success("Connection Established!")
    placeholder1.empty()
    placeholder2.empty()
    placeholder3.empty()
    placeholder4.empty()
    placeholder5.empty()
    placeholder6.empty()

    columns_list = df.columns
    columns_list.insert(0, "Select One")

    model_type = placeholder1.selectbox("Select the Problem Type", ["Select One", "Regression", "Classification"])
    models_list = ["Select One"]
    if model_type == "Regression":
        models_list.extend(["Random Forest Regressor", "Xgboost Regressor", "Linear Regression", "SVM Regression"])
    elif model_type == "Classification": 
        models_list.extend(["Random Forest Classifier", "Xgboost Classifier", "SVM Classification"])

    label_column = placeholder2.selectbox("Select the Label Column", columns_list)
    clustering_bool = placeholder3.selectbox("Do clustering?", ["Select One", "True", "False"])

    list_of_models = []
    if model_type:
        model1 = placeholder4.selectbox("Select First Model", models_list)
        if model1 != "Select One":
            model2 = placeholder5.selectbox("Select Second Model", [model for model in models_list if model != model1])
            if model2 != "Select One":
                model3 = placeholder6.selectbox("Select Third Model", [model for model in models_list if model != model1 and model != model2])
                if model3 != "Select One":
                    list_of_models = [model1,model2,model3]
                    placeholder_for_info.success("Input Taken!")
    if len(list_of_models) != 0:

        placeholder1.empty()
        placeholder2.empty()
        placeholder3.empty()
        placeholder4.empty()
        placeholder5.empty()
        placeholder6.empty()

        if clustering_bool == "True":
            placeholder2.header("Clustering")
            preprocessed_dataset, transform_dict, selected_features = Process(dataset=df, model_type=model_type, label_column=label_column, 
                                                        clustering_bool=clustering_bool).data_preprocessing()
            
            clustering, number_of_clusters = ClusteringDataset(dataset=preprocessed_dataset, label_column=label_column).make_clusters()
            placeholder3.write(f"Optimal Number of clusters: {number_of_clusters}")
            if model_type == "Regression":
                divided_dataset, standard_scaler, min_max_scaler = DivideDatasets(dataset=preprocessed_dataset, number_of_clusters=number_of_clusters, 
                                                                                label_column=label_column, model_type=model_type, clustering=clustering).divide()
                models = MakeClusterModels(datasets=divided_dataset, model_type=model_type, min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)

            elif model_type == "Classification":
                divided_dataset, standard_scaler = DivideDatasets(dataset=preprocessed_dataset, number_of_clusters=number_of_clusters, 
                                                                                label_column=label_column, model_type=model_type, clustering=clustering).divide()

                models = MakeClusterModels(datasets=divided_dataset, model_type=model_type).model(list_of_models=list_of_models)

        elif clustering_bool == "False":
            placeholder2.header("Basic Model Building")
            if model_type == "Regression":
                preprocessed_dataset, transform_dict, standard_scaler, min_max_scaler, selected_features = Process(dataset=df, model_type=model_type, label_column=label_column, 
                                                                                                clustering_bool=clustering_bool).data_preprocessing()
 
                models = MakeBasicModels(dataset=preprocessed_dataset, model_type=model_type, label_column=label_column, 
                                        min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)
            elif model_type == "Classification":
                preprocessed_dataset, transform_dict, standard_scaler, selected_features = Process(dataset=df, model_type=model_type, label_column=label_column, 
                                                                                                clustering_bool=clustering_bool).data_preprocessing()
                
                models, fig = MakeBasicModels(dataset=preprocessed_dataset, model_type=model_type, label_column=label_column, 
                                        ).model(list_of_models=list_of_models)
                placeholder4.pyplot(fig)
            if st.sidebar.button("Save Models"):
                with open("Model.pkl", "wb") as f:
                    pickle.dump(models, f)  
                st.sidebar.success("Model Saved")                   
        placeholder_for_info.success("Models created successfully, upload file for predictions")
        prediction_file = placeholder1.file_uploader("Upload your CSV file", type=["csv"], key="prediction_uploader")

        if prediction_file:
            placeholder1.empty()
            placeholder2.empty()
            placeholder3.empty()
            placeholder4.empty()
            file_details = {"filename":prediction_file.name, "filetype":prediction_file.type,
                            "filesize":prediction_file.size}
            prediction_file = pd.read_csv(prediction_file)
            st.write(file_details)
            if model_type == "Regression":
                if clustering_bool == "True":
                    predicted_df = MakePredictions(data_for_prediction=prediction_file, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                            min_max_scaler=min_max_scaler, models=models, model_type=model_type, clustering=clustering, selected_features=selected_features).predictions()

                elif clustering_bool == "False":                        
                    predicted_df = MakePredictions(data_for_prediction=prediction_file, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                            min_max_scaler=min_max_scaler, models=models, model_type=model_type, selected_features=selected_features).predictions()
            elif model_type == "Classification":
                if clustering_bool == "True":
                    predicted_df = MakePredictions(data_for_prediction=prediction_file, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                            models=models, model_type=model_type, clustering=clustering, selected_features=selected_features).predictions()

                elif clustering_bool == "False":                        
                    predicted_df = MakePredictions(data_for_prediction=prediction_file, transform_dict=transform_dict, standard_scaler=standard_scaler, 
                                            models=models, model_type=model_type, selected_features=selected_features).predictions()
            placeholder_for_info.success("Predictions done on the dataset!")
            st.write(predicted_df)
