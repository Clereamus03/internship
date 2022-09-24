from kneed import KneeLocator
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

class ClusteringDataset:

    def __init__(self, dataset, label_column):
        self.dataset = pd.DataFrame(dataset)
        self.label_column = label_column

    def distortions(self):

        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.dataset.drop([self.label_column], axis=1))
            distortions.append(kmeanModel.inertia_)
        
        return K, distortions

    def select_k_value(self, K, distortions):
    
        kn = KneeLocator(K, distortions, curve='convex', direction='decreasing')
        return kn.knee

    def make_clusters(self):

        ''' 
        This function will make use of above function 
        SelectKValue
        '''
        K, distortions = self.distortions()
        number_of_clusters = self.select_k_value(K, distortions)
        clustering = KMeans(n_clusters=number_of_clusters, random_state=42)
        training_set = self.dataset.drop([self.label_column], axis=1)
        clustering.fit(training_set)
        print("Done Clustering")

        # with open("files/clustering.pkl", "wb") as f:
        #     pickle.dump(clustering, f)

        return clustering, number_of_clusters
