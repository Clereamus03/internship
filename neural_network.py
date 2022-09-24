import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from keras import optimizers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import Hyperband
from keras import callbacks


class NeuralNetwork:

    def __init__(self, x_train, x_test, y_train, y_test, model_type):

        self.X_train = x_train
        self.X_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_type = model_type

    def build_model_regression(self, hp):

        model = tf.keras.Sequential()

        for i in range(hp.Int('num_layers', 2, 11)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                         min_value=32,
                                                         max_value=512,
                                                         step=32),
                                            activation=hp.Choice("activation", ["relu", "linear", "elu"])))
        model.add(tf.keras.layers.Dense(1, activation=hp.Choice("activation", ["relu", "linear", "elu"])))

        model.compile(optimizer=tf.keras.optimizers.Adam
        (hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='mean_absolute_error',
                      metrics=['mean_absolute_error'])

        return model

    def build_model_classification(self, hp):

        model = tf.keras.Sequential()

        for i in range(hp.Int('num_layers', 2, 11)):
            model.add(tf.keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                         min_value=32,
                                                         max_value=512,
                                                         step=32),
                                            activation=hp.Choice("activation", ["sigmoid", "relu", "softmax"])))
        model.add(tf.keras.layers.Dense(1, activation=hp.Choice("activation", ["sigmoid", "relu", "softmax"])))
        model.compile(optimizer=tf.keras.optimizers.Adam
        (hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

    def hp_tuning(self):

        build_model = ''
        obj = ''
        dir = ''

        if self.model_type == "Regression":
            obj = 'val_mean_absolute_error'
            build_model = self.build_model_regression
            dir = 'house_data'

        elif self.model_type == "Classification":
            obj = 'val_accuracy'
            build_model = self.build_model_classification
            dir = 'clf_data'

        # HyperBand algorithm from keras tuner

        tuner = Hyperband(build_model,
                          objective=obj,
                          max_epochs=25,
                          executions_per_trial=4,
                          directory=dir)

        return tuner

    def main(self):

        tuner = self.hp_tuning()

        tuner.search(self.X_train, self.y_train,
                     epochs=5,
                     validation_data=(self.X_test, self.y_test))

        model = tuner.get_best_models()[0]

        mon = ''
        mod = ''

        if self.model_type == 'Regression':
            mon = "val_mean_absolute_error"
            mod = "min"
        elif self.model_type == "Classification":
            mon = "val_accuracy"
            mod = "max"

        early_stopping = callbacks.EarlyStopping(monitor=mon,
                                                 mode=mod,
                                                 patience=5,
                                                 restore_best_weights=True)

        model.fit(self.X_train, self.y_train, epochs=100, validation_data=(self.X_test, self.y_test),
                  callbacks=[early_stopping])

        print(model.summary())

        return model
