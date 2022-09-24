import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from ModelTuner import ModelTuning
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

class MakeBasicModels:
    '''
        Description: Prepare different models based on list_of_models
        Param dataset : The dataframe object
        Param model_type: Type of problem statement
        Param min_max_scaler : For inverse scaling of dependent column in Regression
        Param label_column : The dependent column
        Function to run: Model
    '''
    def __init__(self, dataset, model_type, opt_model, label_column, min_max_scaler=None):
        self.dataset = dataset
        self.model_type = model_type
        self.opt_model = opt_model
        self.min_max_scaler = min_max_scaler
        self.label_column = label_column

    def mae(self, actual, predicted):

        from sklearn.metrics import mean_absolute_percentage_error
        return mean_absolute_percentage_error(actual, predicted)

    def mape(self, actual, predicted):

        from sklearn.metrics import mean_absolute_percentage_error
        return mean_absolute_percentage_error(actual, predicted)

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
        rf = RandomForestRegressor(random_state=42, )
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
        classifier = SVC(probability=True)
        classifier.fit(X_train, y_train)
        return classifier

    def bayesian_ridge_regressor(self, X_train, y_train):

        from sklearn import linear_model
        reg = linear_model.BayesianRidge()
        reg.fit(X_train, y_train)
        return reg

    def Quantile_regressor(self, X_train, y_train):

        from sklearn import linear_model
        qr = linear_model.QuantileRegressor(quantile=0.8)
        qr.fit(X_train, y_train)
        return qr

    def Naive_Bayes_classifier(self, X_train, y_train):

        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        return gnb

    def logistic_regression_classifier(self, X_train, y_train):

        from sklearn.linear_model import LogisticRegression
        lrclf = LogisticRegression()
        lrclf.fit(X_train, y_train)
        return lrclf

    def model(self, list_of_models):
        """
            Method Name : Model
            param list_of_models: list of models selected by the user.
            Description : This function runs after object has been created and will
                          make models from list_of_models and checks accuracy.
            returns: model with best accuracy or with the least error
            written by : Nitin 
            version : 1.0
            Revision : None
        """

        # if model to be optimized: Make another models_reference and call function from ModelTuner

        '''opt_model_reference = ["Random Forest Regressor", "Random Forest Classifier", "Xgboost Regressor", "Xgboost Classifier",
                               "Linear Regression", "SVM Regression", "SVM Classification", "Neural Network",
                               "Bayesian Ridge Regression", "NB Classification", "LR Classification"] '''

        models_reference = {"Random Forest Regressor": self.rf_regressor, "Random Forest Classifier": self.rf_classifier, 
                            "Xgboost Regressor": self.xgb_regressor, "Xgboost Classifier": self.xgb_classifier,
                            "Linear Regression": self.linear_regressor, "SVM Regression": self.support_vector_regressor,
                            "SVM Classification": self.support_vector_classifier, "Neural Network": None,
                            "Bayesian Ridge Regression": self.bayesian_ridge_regressor, "NB Classification": self.Naive_Bayes_classifier,
                            "LR Classification": self.logistic_regression_classifier}

        accuracy_list = []
        error_list = []
        trained_models = []
        X_train, X_test, y_train, y_test = self.dataset
        if self.model_type == "Regression":
            y_test = self.min_max_scaler.inverse_transform(np.array(y_test).reshape(-1,1))
        
        elif self.model_type == "Classification":
            from sklearn.metrics import roc_curve, roc_auc_score, plot_confusion_matrix
            r_probs = [0 for i in range(len(y_test))]
            model_probs_list = []

        if self.opt_model == True:
            for model_refs in list_of_models:
                opt_obj = ModelTuning(self.model_type, model_refs, X_train, y_train, X_test, y_test)
                model = opt_obj.main()

                trained_models.append(model)

                if self.model_type == "Regression":
                    model_predictions = model.predict(X_test)
                    model_predictions = self.min_max_scaler.inverse_transform(np.array(model_predictions).reshape(-1,1))
                    mae = self.mae(actual= y_test, predicted = model_predictions)
                    mae = np.array(mae).reshape(-1,1)
                    print(f"{i} Model has an mean absolute error of {mae}")
                    error_list.append(mae)

                else:
                    model_predictions = model.predict(X_test)
                    model_probs = model.predict_proba(X_test)
                    model_probs = model_probs[:, 1]
                    model_probs_list.append(model_probs)
                    tp, fn, fp, tn = self.confusion_matrix(actual=y_test, predicted=model_predictions)
                    acc = self.accuracy_score_classification(actual=y_test, predicted=model_predictions)
                    print(f"{i} has an accuracy of {acc}")
                    accuracy_list.append(acc)

        else:
            for i in list_of_models:
                # IF NN :
                if i == "Neural Network":
                    nn_object = NeuralNetwork(X_train, X_test, y_train, y_test, self.model_type)
                    model = nn_object.main()
                else:
                    model = models_reference[i](X_train = X_train, y_train = y_train.values.ravel())

                trained_models.append(model)

                if self.model_type == "Regression":
                    model_predictions = model.predict(X_test)
                    model_predictions = self.min_max_scaler.inverse_transform(np.array(model_predictions).reshape(-1,1))
                    mae = self.mae(actual= y_test, predicted = model_predictions)
                    mae = np.array(mae).reshape(-1,1)
                    print(f"{i} Model has an mean absolute error of {mae}")
                    error_list.append(mae)
                else:
                    if i == "Neural Network":
                        model_probs = model.predict(X_test)
                        model_predictions = np.round_(model_probs)
                        # model_probs = model_probs[:, 1]
                        model_probs_list.append(model_probs)
                        tp, fn, fp, tn = self.confusion_matrix(actual=y_test, predicted=model_predictions)
                        acc = self.accuracy_score_classification(actual=y_test, predicted=model_predictions)
                        print(f"{i} has an accuracy of {acc}")
                        accuracy_list.append(acc)
                    else:
                        model_predictions = model.predict(X_test)
                        model_probs = model.predict_proba(X_test)
                        model_probs = model_probs[:, 1]
                        model_probs_list.append(model_probs)
                        tp, fn, fp, tn = self.confusion_matrix(actual= y_test, predicted = model_predictions)
                        acc = self.accuracy_score_classification(actual= y_test, predicted = model_predictions)
                        print(f"{i} has an accuracy of {acc}")
                        accuracy_list.append(acc)

        if self.model_type == "Regression":
            [[mae_of_least_error_model]] = min(error_list)
            st.write(f"Model: {list_of_models[error_list.index(min(error_list))]}")
            st.write(f"Mean Absolute Error: {mae_of_least_error_model}")
            return trained_models[error_list.index(min(error_list))]
        elif self.model_type == "Classification":
            
            st.set_option('deprecation.showPyplotGlobalUse', False)
            auc_score = []
            roc_fpr = []
            roc_tpr = []
            r_auc = roc_auc_score(y_test, r_probs)

            r_fpr, r_tpr, _ = roc_curve(y_test, r_probs)
            for i in range(len(model_probs_list)):
                
                auc_score.append(roc_auc_score(y_test, model_probs_list[i]))
                fpr, tpr, _ = roc_curve(y_test, model_probs_list[i])
                roc_fpr.append(fpr)
                roc_tpr.append(tpr)
            
            fig = plt.figure(figsize=(15,10))
            plt.plot(r_fpr, r_tpr, linestyle='--', label='Random prediction (AUROC = %0.3f)' % r_auc)
            for i in range(len(list_of_models)):
                plt.plot(roc_fpr[i], roc_tpr[i], marker='.', label=f'{list_of_models[i]} (AUROC = {auc_score[i]})')

            plt.title('ROC Plot')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(fontsize=12, loc="lower right")

            # Plotting Confusion Matrix
            # cm = confusion_matrix(y_true = y_test, y_pred = model_predictions)
            # cm = confusion_matrix(y_test, model_predictions)
            # f = sns.heatmap(cm, annot=True)

            st.write(f"Model: {list_of_models[accuracy_list.index(max(accuracy_list))]}")
            st.write(f"Accuracy: {max(accuracy_list)} %")
            return trained_models[accuracy_list.index(max(accuracy_list))], X_test, y_test, fig, model_predictions