import pandas as pd
import numpy as np
import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
# from xgboost import XGBClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
# from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import optuna


class ModelTuning:

    def __init__(self, model_type, model_reference, X_train, y_train, X_test, y_test):

        self.model_type = model_type
        self.model_reference = model_reference
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.rg_tuner = self.ModelTuningRegressor(model_reference, X_train, y_train, X_test, y_test)
        self.clf_tuner = self.ModelTuningClassifier(model_reference, X_train, y_train)

    class ModelTuningRegressor:

        def __init__(self, model_reference, X_train, y_train, X_test, y_test):

            self.model_reference = model_reference
            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

        def objective_rf_regressor(self, trial):

            # Defining parameters to be tuned
            n_estimators = trial.suggest_int('n_estimators', 30, 1000)
            max_depth = trial.suggest_int('max_depth', 1, 10000)
            criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'poisson'])
            max_features = trial.suggest_categorical('max_features', ['None', 'sqrt', 'log2'])

            rg = sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                        criterion=criterion, max_features=max_features)
            rg.fit(self.X_train, self.y_train)

            # Evaluating score using cv
            return mean_squared_error(self.y_test, rg.predict(self.X_test))

        def objective_xgb_regressor(self, trial):

            # Defining parameters to be tuned
            n_estimators = trial.suggest_int('n_estimators', 1, 1000)
            max_depth = trial.suggest_int('max_depth', 1, 100)
            eta = trial.suggest_loguniform('eta', 1e-8, 1.0)
            subsample = trial.suggest_categorical('subsample', [0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
            colsample_bytree = trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

            rg = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, eta=eta, subsample=subsample,
                              colsample_bytree=colsample_bytree)
            rg.fit(self.X_train, self.y_train)

            # Evaluating score using cv
            return mean_squared_error(self.y_test, rg.predict(self.X_test))

        def objective_linear_regressor(self, trial):

            # Defining parameters to be tuned
            fit_intercept = trial.suggest_categorical('fit_intercept', ['True', 'False'])
            positive = trial.suggest_categorical('positive', ['True', 'False'])

            rg = LinearRegression(fit_intercept=fit_intercept, positive=positive)
            rg.fit(self.X_train, self.y_train)

            # Evaluating score using cv
            return mean_squared_error(self.y_test, rg.predict(self.X_test))

        def objective_svm_regressor(self, trial):

            # Defining parameters to be tuned
            kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
            C = trial.suggest_float('C', 0.1, 100.0, log=True)
            gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
            degree = trial.suggest_int('degree', 1, 3, log=True)

            rg = SVR(kernel=kernel, C=C, gamma=gamma, degree=degree)
            rg.fit(self.X_train, self.y_train)

            # Evaluating score using cv
            return mean_squared_error(self.y_test, rg.predict(self.X_test))

        def get_params(self):

            # Creating a dicionary to store all models and its objective function.
            switcher = {"Random Forest Regressor": self.objective_rf_regressor,
                        "Xgboost Regressor": self.objective_xgb_regressor,
                        "Linear Regression": self.objective_linear_regressor,
                        "SVM Regression": self.objective_svm_regressor}

            def switch():
                return switcher.get(self.model_reference)

            obj = switch()

            study = optuna.create_study(direction='minimize')
            study.optimize(obj, n_trials=5)
            trial = study.best_trial

            return trial

        def main_rg(self):

            # Get best params
            trial = self.get_params()

            # Creating model using best params
            if self.model_reference == "Random Forest Regressor":
                model = sklearn.ensemble.RandomForestRegressor(n_estimators=trial.params['n_estimators'],
                                                               max_depth=trial.params['max_depth'],
                                                               criterion=trial.params['criterion'],
                                                               max_features=trial.params['max_features'])

            elif self.model_reference == "Xgboost Regressor":
                model = XGBRegressor(n_estimators=trial.params['n_estimators'],
                                     max_depth=trial.params['max_depth'],
                                     eta=trial.params['eta'],
                                     subsample=trial.params['subsample'],
                                     colsample_bytree=trial.params['colsample_bytree'])

            elif self.model_reference == "Linear Regression":
                model = LinearRegression(fit_intercept=trial.params['fit_intercept'],
                                         positive=trial.params['positive'])

            elif self.model_reference == "SVM Regression":
                model = SVR(kernel=trial.params['kernel'],
                            C=trial.params['C'],
                            gamma=trial.params['gamma'],
                            degree=trial.params['degree'])

            else:
                print("Incorrect model reference given.")

            # Create model using best params
            # rf_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])
            model.fit(X_train, y_train)

            return model

    class ModelTuningClassifier:

        def __init__(self, model_reference, X_train, y_train):

            self.model_reference = model_reference
            self.X_train = X_train
            self.y_train = y_train

        def objective_rf_classifier(self, trial):

            # Defining parameters to be tuned
            n_estimators = trial.suggest_int('n_estimators', 10, 1000)
            max_depth = int(trial.suggest_loguniform('max_depth', 2, 32))
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            max_features = trial.suggest_categorical('max_features', ['None', 'sqrt', 'log2'])

            clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                                          max_depth=max_depth)

            # Evaluating score using cv
            return sklearn.model_selection.cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

        def objective_xgb_classifier(self, trial):

            # Defining parameters to be tuned
            n_estimators = trial.suggest_int('n_estimators', 1, 1000)
            max_depth = trial.suggest_int('max_depth', 2, 25)
            reg_alpha = trial.suggest_int('reg_alpha', 0, 5)
            reg_lambda = trial.suggest_int('reg_lambda', 0, 5)
            min_child_weight = trial.suggest_int('min_child_weight', 0, 5)
            gamma = trial.suggest_int('gamma', 0, 5)
            learning_rate = trial.suggest_loguniform('learning_rate', 0.005, 0.5)

            clf = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, reg_alpha=reg_alpha,
                                reg_labda=reg_lambda,
                                min_child_weight=min_child_weight, gamma=gamma, learning_rate=learning_rate)

            # Evaluating score using cv
            return sklearn.model_selection.cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

        def objective_svm_classifier(self, trial):

            # Defining parameters to be tuned
            kernel = trial.suggest_categorical('kernel', ['rbf', 'poly', 'linear', 'sigmoid'])
            C = trial.suggest_float('C', 0.1, 100, log=True)
            gamma = trial.suggest_categorical('gamma', ['auto', 'scale'])
            degree = trial.suggest_int('degree', 1, 3, log=True)

            clf = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree)

            # Evaluating score using cv
            return sklearn.model_selection.cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

        def objective_nb_classifier(self, trial):

            # Defining parameters to be tuned
            var_smoothing = trial.suggest_float('var_smoothing', 1e-9, 1.0, log=True)

            clf = GaussianNB(priors=None, var_smoothing=var_smoothing)

            # Evaluating score using cv
            return sklearn.model_selection.cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

        def objective_lr_classifier(self, trial):

            # Defining parameters to be tuned
            solver = trial.suggest_categorical('solver', ['newton-cg', 'lbfgs', 'liblinear', 'saga'])
            penalty = trial.suggest_categorical('penalty', ['none', 'l1', 'l2'])
            C = trial.suggest_float('C', 1e-10, 1e10, log=True)

            clf = LogisticRegression(solver=solver, penalty=penalty, C=C)

            # Evaluating score using cv
            return sklearn.model_selection.cross_val_score(clf, self.X_train, self.y_train, n_jobs=-1, cv=3).mean()

        def get_params(self):

            def default():
                return "Incorrect model name"

            # Creating a dicionary to store all models and its objective function.
            switcher = {"Random Forest Classifier": self.objective_rf_classifier,
                        "Xgboost Classifier": self.objective_xgb_classifier,
                        "SVM Classification": self.objective_svm_classifier,
                        "NB Classification": self.objective_nb_classifier,
                        "LR Classification": self.objective_lr_classifier}

            def switch():
                return switcher.get(self.model_reference)

            obj = switch()

            study = optuna.create_study(direction='maximize')
            study.optimize(obj, n_trials=5)
            trial = study.best_trial

            return trial

        def main_clf(self):

            # Get best params
            trial = self.get_params()

            # Creating model using best params
            if self.model_reference == "Random Forest Classifier":
                model = sklearn.ensemble.RandomForestClassifier(n_estimators=trial.params['n_estimators'],
                                                                max_depth=trial.params['max_depth'],
                                                                criterion=trial.params['criterion'])

            elif self.model_reference == "Xgboost Classifier":
                model = XGBClassifier(n_estimators=trial.params['n_estimators'],
                                      max_depth=trial.params['max_depth'],
                                      reg_alpha=trial.params['reg_alpha'],
                                      reg_labda=trial.params['reg_lambda'],
                                      min_child_weight=trial.params['min_child_weight'],
                                      gamma=trial.params['gamma'],
                                      learning_rate=trial.params['learning_rate'])

            elif self.model_reference == "SVM Classification":
                model = SVC(kernel=trial.params['kernel'],
                            C=trial.params['C'],
                            gamma=trial.params['gamma'],
                            degree=trial.params['degree'])

            elif self.model_reference == "NB Classification":
                model = GaussianNB(priors=None,
                                   var_smoothing=trial.params['var_smoothing'])

            elif self.model_reference == "LR Classification":
                model = LogisticRegression(solver=trial.params['solver'],
                                           penalty=trial.params['penalty'],
                                           C=trial.params['C'])

            else:
                print("Incorrect model reference given.")

            # Create model using best params
            # rf_clf = sklearn.ensemble.RandomForestClassifier(n_estimators=trial.params['n_estimators'], max_depth=trial.params['max_depth'])
            model.fit(X_train, y_train)

            return model

    def main(self):

        if self.model_type == "Regression":
            obj = self.ModelTuningRegressor(model_reference, X_train, y_train, X_test, y_test)
            model = obj.main_rg()

        elif self.model_type == "Classification":
            obj = self.ModelTuningClassifier(model_reference, X_train, y_train)
            model = obj.main_clf()

        return model