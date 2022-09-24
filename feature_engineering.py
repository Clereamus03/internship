import pandas as pd
import streamlit as st
import numpy as np
class Engineering:

    def __init__(self, dataset, label_column):
        self.dataset = dataset
        self.label_column = label_column

    def engineering_with_autofeat(self):
        """
            Method Name : engineering_with_autofeat
            Description : This method deletes all unrelated
                          columns from the dataset.
            returns: list of columns selected using autofeat
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        from autofeat import FeatureSelector
        X = self.dataset.drop([self.label_column], axis=1)
        y = self.dataset[self.label_column]
        fsel = FeatureSelector(verbose=1)
        new_X = fsel.fit_transform(pd.DataFrame(X), pd.DataFrame(y))

        return list(new_X.columns)

    def engineering_with_pps(self):
        """
            Method Name : engineering_with_pps
            Description : This method deletes all unrelated
                          columns from the dataset.
            returns: list of columns selected using ppscore
            written by : Nitin 
            version : 1.0
            Revision : None
        """

        import ppscore as pps

        ppscore_cols = []
        for i in self.dataset.columns:
            if i != self.label_column:
                score = pps.score(self.dataset,i, self.label_column)
                if score['ppscore'] >= 0.1:
                    ppscore_cols.append(score['x'])

        return ppscore_cols

    def feature_selector(self):
        """
            Method Name : feature_selector
            Description : This method uses both autofeat and ppscore
                          to get the best features.
            returns: dataset with selected features 
            written by : Nitin 
            version : 1.0
            Revision : None
        """
        autofeat_cols = self.engineering_with_autofeat()
        print("Autofeat selected cols",autofeat_cols)
        ppscore_cols = self.engineering_with_pps()
        print("PPscore selected cols", ppscore_cols)
        if len(autofeat_cols)>0 and len(ppscore_cols)>0: 
            final_cols = list(set(autofeat_cols + ppscore_cols))
        elif len(ppscore_cols) == 0 and len(autofeat_cols) == 0:
            final_cols = self.dataset.columns
        elif len(autofeat_cols) == 0:
            final_cols = ppscore_cols
        elif len(ppscore_cols) == 0:
            final_cols = autofeat_cols
            
        final_cols = list(set(final_cols))
        dataset_cols = final_cols.copy()
        dataset_cols.append(self.label_column)

        return self.dataset[dataset_cols], final_cols
