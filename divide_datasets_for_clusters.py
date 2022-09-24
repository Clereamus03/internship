import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle

class DivideDatasets:

    def __init__(self, dataset, number_of_clusters, label_column, model_type, clustering):

        self.dataset = dataset
        self.number_of_clusters = number_of_clusters
        self.label_column = label_column
        self.model_type = model_type
        self.clustering = clustering

    def dataset_train_test_split(self):
        training_set = self.dataset.drop([self.label_column], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(training_set, self.dataset[self.label_column], test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train=None, y_test=None):
        """
            Method Name : ScaleFeatures
            Description : This method converts independent variables
                          using Standard Scaling and dependent variable
                          (Regression) using Min Max Scaling.
            param X_train: Independent Train Variables
            param X_test: Independent Test Variables
            param y_train: Dependent Train Variables(default=None)
            param y_test: Dependent Test Variables(default=None)
            return: dataset with only numerical values
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        if self.model_type == "Regression":

            y_train = np.array(y_train).reshape(-1, 1)
            y_test = np.array(y_test).reshape(-1,1)

            min_max_scaler = preprocessing.MinMaxScaler()
            
            y_train = min_max_scaler.fit_transform(y_train)
            y_test = min_max_scaler.transform(y_test)

            # with open("files/min_max_scaler.pkl", "wb") as f:
            #     pickle.dump(min_max_scaler, f)
        
        standard_scaler = preprocessing.StandardScaler()
        
        to_scale = [col for col in X_train.columns.values]
        standard_scaler.fit(X_train[to_scale])
        X_train[to_scale] = standard_scaler.transform(X_train[to_scale])
        X_test[to_scale] = standard_scaler.transform(X_test[to_scale])

        # with open("files/standard_scaler.pkl", "wb") as f:
        #     pickle.dump(standard_scaler, f)

        if self.model_type == "Regression":
            data = [pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)]
            return data, standard_scaler, min_max_scaler
        
        else:
            return pd.DataFrame(X_train), pd.DataFrame(X_test), standard_scaler

    def assign_values(self, X_train, X_test, y_train, y_test):

        train_labels = self.clustering.predict(X_train)
        test_labels = self.clustering.predict(X_test)

        if self.model_type == "Regression":
            data, standard_scaler, min_max_scaler = self.scale_features(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
            X_train, X_test, y_train, y_test = data
        else:
            X_train, X_test, standard_scaler = self.scale_features(X_train=X_train, X_test=X_test)

        X_train_clusters = X_train.copy()
        X_test_clusters = X_test.copy()

        X_train_clusters['clusters'] = train_labels
        X_test_clusters['clusters'] = test_labels
        
        X_train_clusters['y'] = np.array(y_train.copy())
        X_test_clusters['y']  = np.array(y_test.copy())
        
        if self.model_type == "Regression":
            return X_train_clusters, X_test_clusters, standard_scaler, min_max_scaler
        else:
            return X_train_clusters, X_test_clusters, standard_scaler

    def divide(self):

        X_train, X_test, y_train, y_test = self.dataset_train_test_split()
        if self.model_type == "Regression":
            train_clusters, test_clusters, standard_scaler, min_max_scaler = self.assign_values(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        else:
            train_clusters, test_clusters, standard_scaler = self.assign_values(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        
        datasets = {}
        for i in range(self.number_of_clusters):
            train_local = train_clusters.loc[train_clusters.clusters == i]
            y_train_local = train_local.y.values
            train_local = train_local.drop(columns=['y', 'clusters'], axis=1)
            
            test_local = test_clusters.loc[test_clusters.clusters == i]
            y_test_local = test_local.y.values
            test_local = test_local.drop(columns=['y', 'clusters'], axis=1)

            cluster_number = f"cluster-{i}"
            datasets[cluster_number] = [train_local, test_local, y_train_local, y_test_local]

        print(f"Divided the dataset in {self.number_of_clusters} clusters")
        if self.model_type == "Regression":
            return datasets, standard_scaler, min_max_scaler
        else:
            return datasets, standard_scaler

    