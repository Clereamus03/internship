from distutils.log import error
import numpy as np
import streamlit as st
class MakeClusterModels:
    '''
        Description: Prepare different models based on list_of_models
        Param dataset : The dataframe object
        Param model_type: Type of problem statement
        Param min_max_scaler : For inverse scaling of dependent column in Regression
        Param label_column : The dependent column
        Function to run: Model
    '''
    def __init__(self, datasets, model_type, min_max_scaler=None):
        self.datasets = datasets
        self.model_type = model_type
        self.min_max_scaler = min_max_scaler

    def mae(self, actual, predicted):

        from sklearn.metrics import mean_absolute_error   
        return mean_absolute_error(actual, predicted)

    def accuracy_score_classification(self, actual, predicted):

        from sklearn.metrics import accuracy_score
        acc = (accuracy_score(actual, predicted)) * 100 
        return acc

    def confusion_matrix(self, actual, predicted):

        from sklearn.metrics import confusion_matrix
    
        tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
        return tp, fn, fp, tn

    def xgb_regressor(self, X_train, y_train):

        import xgboost as xgb
        xgb.set_config(verbosity=0)
        xg = xgb.XGBRegressor()
        xg.fit(X_train, y_train)
        return xg

    def xgb_classifier(self, X_train, y_train):

        import xgboost as xgb
        xgb.set_config(verbosity=0)
        xg = xgb.XGBClassifier()
        xg.fit(X_train, y_train)
        return xg

    def rf_regressor(self, X_train, y_train):

        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(random_state=42)
        rf.fit(X_train, y_train)
        return rf

    def rf_classifier(self, X_train, y_train):

        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        return rf

    def linear_regressor(self, X_train, y_train):

        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        return lr 

    def support_vector_regressor(self, X_train, y_train):

        from sklearn.svm import SVR
        regressor = SVR(kernel = 'rbf')  # Kernel optional or default??
        regressor.fit(X_train, y_train)
        return regressor

    def support_vector_classifier(self, X_train, y_train):

        from sklearn.svm import SVC
        classifier = SVC()
        classifier.fit(X_train, y_train)
        return classifier

    def model(self, list_of_models):
        """
            Method Name : Model
            param list_of_models: list of models selected by the user.
            Description : This function runs after object has been created and will
                          make models for each cluster from list_of_models and checks accuracy.
            returns: models for each cluster with best accuracy or with the least error
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        models_reference = {"Random Forest Regressor": self.rf_regressor, "Random Forest Classifier": self.rf_classifier, 
                            "Xgboost Regressor": self.xgb_regressor, "Xgboost Classifier": self.xgb_classifier,
                            "Linear Regression": self.linear_regressor, "SVM Regression": self.support_vector_regressor,
                            "SVM Classification": self.support_vector_classifier}

        final_models = {}
        selected_models_name = {}
        selected_models_metrics_score = []
        for i in range(len(self.datasets.keys())):
            
            cluster = f"cluster-{i}"
            errors_list = []
            models_list = []

            if self.model_type =="Regression":
                
                for j in list_of_models:
                    model = models_reference[j](X_train = self.datasets[cluster][0], y_train =self.datasets[cluster][2])
                    y_test = self.min_max_scaler.inverse_transform(np.array(self.datasets[cluster][3]).reshape(-1,1))
                    model_predictions = model.predict(self.datasets[cluster][1])
                    model_predictions = self.min_max_scaler.inverse_transform(np.array(model_predictions).reshape(-1,1))
                    mae = self.mae(actual=y_test, predicted=model_predictions)
                    mae = np.array(mae).reshape(-1,1)

                    errors_list.append(mae)
                    models_list.append(model)

                least_error_model = models_list[errors_list.index(min(errors_list))]
                final_models[cluster] = least_error_model
                [[mae_of_least_error_model]] = min(errors_list)
                # st.write(f"Model: {least_error_model} with Mean Absolute Error: {mae_of_least_error_model} \n")
                selected_models_name[cluster] = list_of_models[errors_list.index(min(errors_list))]
                selected_models_metrics_score.append(mae_of_least_error_model)

            else:

                for j in list_of_models:

                    model = models_reference[j](X_train = self.datasets[cluster][0], y_train =self.datasets[cluster][2])

                    model_predictions = model.predict(self.datasets[cluster][1])

                    tp, fn, fp, tn = self.confusion_matrix(actual= self.datasets[cluster][3], predicted = model_predictions)

                    errors_list.append(tp+tn)
                    models_list.append(model)

                least_error_model = models_list[errors_list.index(max(errors_list))]
                final_models[cluster] = least_error_model
                acc = self.accuracy_score_classification(actual=self.datasets[cluster][3], predicted=least_error_model.predict(self.datasets[cluster][1]))
                # st.write(f"Model: {least_error_model} with  accuracy: {acc}\n")
                selected_models_name[cluster] = list_of_models[errors_list.index(min(errors_list))]
                selected_models_metrics_score.append(acc)

            print(errors_list)
            print(models_list)
            
        st.write("Models Taken:")
        st.write(selected_models_name)
        ui_accuracy = sum(selected_models_metrics_score)/len(selected_models_metrics_score)
        if self.model_type == "Regression":
            st.write(f"Mean Absolute Error: {ui_accuracy}")
        elif self.model_type == "Classification":
            st.write(f"Accuracy: {ui_accuracy} %")
        return final_models