# Imputing the values - 
#       Filling the missing data with the mean or median value if it’s a numerical variable.
#       Filling the missing data with mode if it’s a categorical value.
#       Filling the numerical value with 0 or -999, or some other number that will not occur in the data. This can be done so that the machine can recognize that the data is not real or is different.
#       Filling the categorical value with a new type for the missing values.
#    Use the fillna() function to fill the null values in the dataset.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from feature_engineering import Engineering

class Process:
    '''
        Description: Basic Pre-Processing Steps 
        Param dataset : The dataframe object
        Param model_type: Type of problem statement
        returns : Processed dataset
    '''
    def __init__(self, dataset, model_type, label_column, clustering_bool):
        self.dataset = dataset
        self.model_type = model_type
        self.label_column = label_column
        self.clustering_bool = clustering_bool

    def impute_nan(self):
        """
            Method Name : impute_nan
            Description : This method checks for the nan values
                          and do the imputation using KNN
            return: dataset with zero nan values
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        if self.dataset[self.label_column].isnull().any():
            print("Null values in the target column")
        else:
            numerical_with_nan = [feature for feature in self.dataset.columns if self.dataset[feature].isnull().any() 
                                and self.dataset[feature].dtypes != "O"]
            categorical_with_nan = [feature for feature in self.dataset.columns if self.dataset[feature].isnull().any() 
                                    and self.dataset[feature].dtypes == "O"]
            if len(numerical_with_nan)>0 or len(categorical_with_nan)>0:
                for feature in numerical_with_nan:
                    print(f"{feature}: {np.round(self.dataset[feature].isnull().mean(),4)} % missing values")
                
                for feature in categorical_with_nan:
                    print(f"{feature}: {np.round(self.dataset[feature].isnull().mean(),4)} % missing values")
                
                # Removing the columns with more than 15% nan values
                pct_null = self.dataset.isnull().sum() / len(self.dataset)
                missing_features = pct_null[pct_null > 0.15].index
                self.dataset.drop(missing_features, axis=1, inplace=True)
                
                # Can tune the value of neighbours in KNNImputer ( default = 5 )
                imputer = KNNImputer()
                after_imputation = imputer.fit_transform(self.dataset)   
                self.dataset = pd.DataFrame(after_imputation,columns=self.dataset.columns)

            else:
                print("There are no Nan values in your dataset")
            
    def transform_categories(self):
        """
            Method Name : transform_categories
            Description : This method converts all categorical
                          values to numbers using transform_dict
            return: dataset with only numerical values
            written by : Nitin 
            version : 1.0
            Revision : None
        """

        lst = []
        transform_dict = {}

        # remove column with more than 30 categories in it.
        for column in self.dataset.columns:
            if self.dataset[column].dtype == "O" and self.dataset[column].name != self.label_column:
                if self.dataset[column].value_counts().count() > 30:
                    self.dataset.drop([column], axis=1, inplace=True)
                else:
                    lst.append(self.dataset[column].name)    

        # This will add Categorical columns with their nested dictionary of categories with their labels
        for col in lst:
            cats = pd.Categorical(self.dataset[col]).categories
            d = {}
            for i, cat in enumerate(cats):
                d[cat] = i
            transform_dict[col] = d
            
        self.dataset = self.dataset.replace(transform_dict)
        
        # with open("files/encoder.pkl", "wb") as f:
        #     pickle.dump(transform_dict, f)
        
        return transform_dict

    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train=None, y_test=None):
        """
            Method Name : scale_features
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
            return [pd.DataFrame(X_train), pd.DataFrame(X_test), pd.DataFrame(y_train), pd.DataFrame(y_test)], standard_scaler, min_max_scaler
        
        else:
            return pd.DataFrame(X_train), pd.DataFrame(X_test), standard_scaler

    def outlier_removal(self):
        """            
            Method Name : outlier_removal
            Description : This method removes the outliers from dataset.
            return: dataset with zero outliers.
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        Q1=self.dataset.quantile(0.25)
        Q3=self.dataset.quantile(0.75)
        IQR=Q3-Q1

        self.dataset=self.dataset[~((self.dataset<(Q1-1.5*IQR)) | (self.dataset>(Q3+1.5*IQR))).any(axis=1)]

    def dataset_train_test_split(self):
        """
            Method Name : dataset_train_test_split
            Description : This function splits dataset using train_test_split
            return: the training and test dataset
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        training_set = self.dataset.drop([self.label_column], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(training_set, self.dataset[self.label_column], test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def data_preprocessing(self):
        """
            Method Name : data_preprocessing
            Description : This method will be ran by framework to 
                          run all pre-processing steps
            returns dataset, transform_dict: if clustering_bool==True
            returns data, transform_dict, standard_scaler, min_max_scaler : if clustering_bool==False
                            { data = [X_train, X_test, y_train, y_test] }
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        transform_dict = self.transform_categories()
        self.impute_nan()
        print(self.dataset)
        self.dataset, selected_features = Engineering(dataset=self.dataset, label_column=self.label_column).feature_selector()
        self.outlier_removal()
        print(self.dataset)
        if self.clustering_bool == "True":
            return self.dataset, transform_dict, selected_features
        else:

            X_train, X_test, y_train, y_test = self.dataset_train_test_split()

            if self.model_type != "Regression":
                X_train, X_test, standard_scaler = self.scale_features(X_train=X_train, X_test=X_test)
                data = [X_train, X_test, y_train, y_test]
                return data, transform_dict, standard_scaler, selected_features
            else:
                [X_train, X_test, y_train, y_test], standard_scaler, min_max_scaler = self.scale_features(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
                data = [X_train, X_test, y_train, y_test]
                return data, transform_dict, standard_scaler, min_max_scaler, selected_features
