
import pandas as pd
import numpy as np
import streamlit as st
import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn

class MakePredictions:

    def __init__(self, data_for_prediction, transform_dict, standard_scaler, models, model_type, selected_features=None, min_max_scaler=None, clustering=None):
        self.data_for_prediction = data_for_prediction
        self.transform_dict = transform_dict
        self.standard_scaler = standard_scaler
        self.min_max_scaler = min_max_scaler
        self.models = models
        self.model_type = model_type
        self.clustering = clustering
        self.selected_features = selected_features

    def clustered_prediction(self):
        """
            Method Name : clustered_prediction
            Description : This method repeats the pre-processing steps and
                          attach the predictions to the file uploaded by the user.
            return: uploaded dataset with predictions for visualizations
            written by : Nitin 
            version : 1.0
            Revision : None
        """

        output_list = []

        final_df = self.data_for_prediction.copy()
        self.data_for_prediction = self.data_for_prediction.replace(self.transform_dict)
        self.data_for_prediction = self.data_for_prediction[self.selected_features]
        self.data_for_prediction = pd.DataFrame(self.standard_scaler.transform(self.data_for_prediction), columns=self.selected_features)
        
        for i in range(len(self.data_for_prediction)):
            datapoint = np.array(self.data_for_prediction.reset_index(drop=True).loc[i])
            datapoint = datapoint.reshape(1,-1)

            cluster = self.clustering.predict(datapoint)[0]
        
            prediction = self.models[f"cluster-{cluster}"].predict(datapoint)
            [prediction] = prediction
            output_list.append(prediction)
            
        if self.model_type == "Regression":
            output_list = self.min_max_scaler.inverse_transform(np.array(output_list).reshape(-1,1))
        
        final_df["Predictions"] = output_list
        # final_df.to_csv("Predictions.csv")

        return final_df

    def basic_prediction(self):
        """
            Method Name : basic_prediction
            Description : This method repeats the pre-processing steps and
                          attach the predictions to the file uploaded by the user.
            return: uploaded dataset with predictions for visualizations
            written by : Nitin 
            version : 1.0
            Revision : None
        """

        final_df = self.data_for_prediction.copy()
 
        self.data_for_prediction = self.data_for_prediction.replace(self.transform_dict)
        self.data_for_prediction = self.data_for_prediction[self.selected_features]
        self.data_for_prediction = self.standard_scaler.transform(self.data_for_prediction)
        output_list = self.models.predict(self.data_for_prediction)

        if self.model_type == "Regression":
            output_list = self.min_max_scaler.inverse_transform(np.array(output_list).reshape(-1,1))

        final_df["Predictions"] = output_list
        # final_df.to_csv("Predictions.csv")

        return final_df

    def predictions(self):

        if self.clustering == None:
            final_df = self.basic_prediction()
        else:
            final_df = self.clustered_prediction()

        return final_df