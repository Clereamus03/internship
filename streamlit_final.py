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
import matplotlib.pyplot as plt
import warnings
import streamlit as st
from eda import Preprocess
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)


def warn(*args, **kwargs):
    pass


warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

database_type = st.sidebar.selectbox("Select the Database",
                                     ["Select One", "MongoDB", "Redis", "Postgres", "MySQL", "Upload File"])

df = pd.DataFrame()
placeholder1 = st.empty()
placeholder2 = st.empty()
placeholder3 = st.empty()
placeholder4 = st.empty()
placeholder5 = st.empty()
placeholder6 = st.empty()
placeholder7 = st.empty()
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
                                                      password=password, database_name=database_name,
                                                      collection_name=collection_name)
elif database_type == "Upload File":
    placeholder_for_info.info("Upload the file.")
    dataset = placeholder1.file_uploader("Upload your CSV file", type=["csv"], key="dataset_uploader")
    if dataset:
        df = pd.read_csv(dataset, encoding="latin1")

if df.empty == False:

    placeholder_for_info.success("Connection Established!")
    placeholder1.empty()
    placeholder2.empty()
    placeholder3.empty()
    placeholder4.empty()
    placeholder5.empty()
    placeholder6.empty()
    placeholder7.empty()

    page = st.sidebar.selectbox("Select the task ", ['Modelling', 'EDA'])
    if page == 'Modelling':
        columns_list = df.columns
        columns_list.insert(0, "Select One")
        # label_column
        # if automatic:
        # guess the model_type
        # clustering_bool = "True"
        #
        model_type = placeholder1.selectbox("Select the Problem Type", ["Select One", "Regression", "Classification"])
        models_list = ["Select One"]
        if model_type == "Regression":
            models_list.extend(["Random Forest Regressor", "Xgboost Regressor", "Linear Regression", "SVM Regression", "Bayesian Ridge Regression", "Decision Tree Regression", "Neural Network"])
        elif model_type == "Classification":
            models_list.extend(["Random Forest Classifier", "Xgboost Classifier", "SVM Classification", "NB Classification", "LR Classification", "Neural Network"])

        label_column = placeholder2.selectbox("Select the Label Column", columns_list)
        clustering_bool = placeholder3.selectbox("Do clustering?", ["Select One", "True", "False"])
        opt_model = placeholder4.selectbox("Use optimized models?", ["Select One", "True", "False"])

        list_of_models = []
        if model_type:
            model1 = placeholder5.selectbox("Select First Model", models_list)
            if model1 != "Select One":
                model2 = placeholder6.selectbox("Select Second Model",
                                                [model for model in models_list if model != model1])
                if model2 != "Select One":
                    model3 = placeholder7.selectbox("Select Third Model", [model for model in models_list if
                                                                           model != model1 and model != model2])
                    if model3 != "Select One":
                        list_of_models = [model1, model2, model3]
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
                preprocessed_dataset, transform_dict, selected_features = Process(dataset=df, model_type=model_type,
                                                                                  label_column=label_column,
                                                                                  clustering_bool=clustering_bool).data_preprocessing()

                clustering, number_of_clusters = ClusteringDataset(dataset=preprocessed_dataset,
                                                                   label_column=label_column).make_clusters()
                placeholder3.write(f"Optimal Number of clusters: {number_of_clusters}")
                if model_type == "Regression":
                    divided_dataset, standard_scaler, min_max_scaler = DivideDatasets(dataset=preprocessed_dataset,
                                                                                      number_of_clusters=number_of_clusters,
                                                                                      label_column=label_column,
                                                                                      model_type=model_type,
                                                                                      clustering=clustering).divide()
                    models = MakeClusterModels(datasets=divided_dataset, model_type=model_type, opt_model=opt_model,
                                               min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)

                elif model_type == "Classification":
                    divided_dataset, standard_scaler = DivideDatasets(dataset=preprocessed_dataset,
                                                                      number_of_clusters=number_of_clusters,
                                                                      label_column=label_column, model_type=model_type,
                                                                      clustering=clustering).divide()

                    models = MakeClusterModels(datasets=divided_dataset, model_type=model_type).model(
                        list_of_models=list_of_models)

            elif clustering_bool == "False":
                placeholder2.header("Basic Model Building")
                if model_type == "Regression":
                    preprocessed_dataset, transform_dict, standard_scaler, min_max_scaler, selected_features = Process(
                        dataset=df, model_type=model_type, label_column=label_column,
                        clustering_bool=clustering_bool).data_preprocessing()

                    models = MakeBasicModels(dataset=preprocessed_dataset, model_type=model_type,
                                             label_column=label_column,
                                             min_max_scaler=min_max_scaler).model(list_of_models=list_of_models)
                elif model_type == "Classification":
                    preprocessed_dataset, transform_dict, standard_scaler, selected_features = Process(dataset=df,
                                                                                                       model_type=model_type,
                                                                                                       label_column=label_column,
                                                                                                       clustering_bool=clustering_bool).data_preprocessing()

                    models, X_test, y_test, fig, model_predictions = MakeBasicModels(dataset=preprocessed_dataset, model_type=model_type,
                                                                  label_column=label_column,
                                                                  ).model(list_of_models=list_of_models)
                    #placeholder4.pyplot(fig)
                    placeholder4.pyplot()
                    placeholder5.subheader("Confusion Matrix")
                    # Plotting confusion matrix
                    ax = plt.subplot()
                    cm = confusion_matrix(y_true=y_test, y_pred=model_predictions)
                    cm = confusion_matrix(y_test, model_predictions)
                    sns.heatmap(cm, annot=True, fmt='g', ax=ax);
                    ax.set_xlabel('Predicted labels');
                    ax.set_ylabel('True labels');
                    ax.xaxis.set_ticklabels(['0', '1']);
                    ax.yaxis.set_ticklabels(['0', '1']);
                    placeholder6.pyplot()
                    # plot_confusion_matrix(models, X_test, y_test)
                    # placeholder6.pyplot()
                if st.sidebar.button("Save Models"):
                    with open("Model.pkl", "wb") as f:
                        pickle.dump(models, f)
                    st.sidebar.success("Model Saved")
            placeholder_for_info.success("Models created successfully, upload file for predictions")
            prediction_file = placeholder1.file_uploader("Upload your CSV file", type=["csv"],
                                                         key="prediction_uploader")

            if prediction_file:
                placeholder1.empty()
                placeholder2.empty()
                placeholder3.empty()
                placeholder4.empty()
                placeholder5.empty()
                placeholder6.empty()
                file_details = {"filename": prediction_file.name, "filetype": prediction_file.type,
                                "filesize": prediction_file.size}
                prediction_file = pd.read_csv(prediction_file)
                st.write(file_details)
                if model_type == "Regression":
                    if clustering_bool == "True":
                        predicted_df = MakePredictions(data_for_prediction=prediction_file,
                                                       transform_dict=transform_dict, standard_scaler = standard_scaler,
                                                       min_max_scaler=min_max_scaler, models=models,
                                                       model_type=model_type, clustering=clustering,
                                                       selected_features=selected_features).predictions()

                    elif clustering_bool == "False":
                        predicted_df = MakePredictions(data_for_prediction=prediction_file,
                                                       transform_dict=transform_dict, standard_scaler=standard_scaler,
                                                       min_max_scaler=min_max_scaler, models=models,
                                                       model_type=model_type,
                                                       selected_features=selected_features).predictions()
                elif model_type == "Classification":
                    if clustering_bool == "True":
                        predicted_df = MakePredictions(data_for_prediction=prediction_file,
                                                       transform_dict=transform_dict, standard_scaler=standard_scaler,
                                                       models=models, model_type=model_type, clustering=clustering,
                                                       selected_features=selected_features).predictions()

                    elif clustering_bool == "False":
                        predicted_df = MakePredictions(data_for_prediction=prediction_file,
                                                       transform_dict=transform_dict, standard_scaler=standard_scaler,
                                                       models=models, model_type=model_type,
                                                       selected_features=selected_features).predictions()
                placeholder_for_info.success("Predictions done on the dataset!")
                st.write(predicted_df)

    elif page == 'EDA':

        analyze = st.sidebar.checkbox("Self analysis")
        if not analyze:

            data = df
            # data=Preprocess(data).encoder()
            col = list(data.columns)
            col.insert(0, 'Select one')

            dep_feat = st.selectbox("Which one is your dependent feature?", col)
            if dep_feat != 'Select one':
                imp_feats = Preprocess(data).feature_importance_calculation(dep_feat)
                st.write('Here are the four important features: ', imp_feats)
                t = Preprocess(data).dis_or_cont(dep_feat)
                for i in imp_feats:

                    if Preprocess(data).dis_or_cont(i) == 'dis' and t == 'cont':

                        plt.figure(figsize=(15, 10))
                        plt.boxplot([data[i], data[dep_feat]], notch=True)
                        plt.xlabel([i, dep_feat])
                        plt.ylabel("Values")
                        plt.legend()

                        st.pyplot()
                    elif Preprocess(data).dis_or_cont(i) == 'cont' and t == 'dis':
                        plt.figure(figsize=(15, 10))
                        plt.boxplot([data[i], data[dep_feat]], notch=True)
                        plt.xlabel([i, dep_feat])
                        plt.ylabel("Values")
                        plt.legend()
                        st.pyplot()

                    elif Preprocess(data).dis_or_cont(i) == 'cont' and t == 'cont':

                        plt.figure(figsize=(15, 10))
                        plt.scatter(data[i], data[dep_feat])
                        plt.xlabel(i)
                        plt.ylabel(dep_feat)
                        plt.legend()
                        st.pyplot()

                    elif Preprocess(data).dis_or_cont(i) == 'dis' and t == 'dis':
                        plt.figure(figsize=(15, 10))
                        plt.scatter(data[i], data[dep_feat])
                        plt.xlabel(i)
                        plt.ylabel(dep_feat)
                        plt.legend()
                        st.pyplot()

        if analyze:

            data = df
            ffa = st.sidebar.selectbox("Selection of features for the analysis",
                                       ['Univariate', 'Bivariate', 'Multivariate'])
            if ffa == 'Univariate':

                Preprocess(data).eda_uni()

            elif ffa == "Bivariate":
                st.sidebar.header('Select any two features:')
                feat1 = st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="first")
                feat2 = st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="second")
                Preprocess(data).eda_bi(feat1, feat2)

            elif ffa == 'Multivariate':
                st.sidebar.header("Select three features from the columns of the dataset : ")
                feat1 = st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="first")
                feat2 = st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="second")
                feat3 = st.sidebar.selectbox('select the feature [Anyone]', data.columns, key="third")
                Preprocess(data).eda_multi(feat1, feat2, feat3)
