from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from functools import partial
from scipy.spatial.distance import minkowski
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.feature_selection import f_classif

class KnnFeatureImportance(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class) #initialize the parameters for the counterfactual generator from the parent class
        #initialize the parameters for the counterfactual generator including normalizer and pca 
        self.nbrs = None
        self.data_class = self.all_data[self.model.predict(self.all_data.drop(self.config["target"], axis=1)) == self.target_class]


    def pre_process(self):
        weights = []
        weights_fvalue = self.finding_feature_improtance_using_anova_fvalue(self.data_class)
        for feature in self.data_class.drop(self.config["target"], axis=1).columns:
            if feature in self.config["features_to_change"]:
                weights_feature = weights_fvalue[weights_fvalue["features"] == feature]["f_value"].values[0]
                weights.append(weights_feature)
            else:
                weights.append(1)
        self.knn = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric=partial(minkowski, p=2, w=weights)).fit(self.data_class.drop(self.config["target"], axis=1))
        
    def generate_single_counterfactual(self, obs):
        self.nbrs = self.knn.kneighbors(obs.drop(self.config["target"], axis=1), return_distance=False) 
        for indice in self.nbrs[0]:
            counterfactual = obs.copy()
            for feature in self.config["features_to_change"]:
                counterfactual[feature] = self.data_class.iloc[indice][feature]
            if self.model.predict(counterfactual.drop(self.config["target"], axis=1)) == self.target_class:
                return counterfactual 
        return None

    def generate_mulitple_counterfactuals(self, data_to_genrate):
        """
        This function is used to generate multiple counterfactuals for multiple instances.
        This function get as input a dataframe with multiple instances and generate a counterfactual for each instance that is not in the desired class.
        """
        counterfactual_list = {}
        data_to_genrate = data_to_genrate[self.model.predict(data_to_genrate.drop(self.config["target"], axis=1)) != self.target_class]
        for _, obs in data_to_genrate.iterrows(): 
            obs = obs.to_frame().T
            counterfactual = self.generate_single_counterfactual(obs.copy())
            if counterfactual is not None:
                counterfactual_list[obs["Loan ID"]] = counterfactual
        return counterfactual_list
    
    def finding_feature_improtance_using_anova_fvalue(self, data):
        #finding the most important features using anova f value for each observation 
        f_value, _ = f_classif(data[self.config["features_to_change"]] , data[self.config["target"]])
        f_value = f_value/np.sum(f_value)
        f_value = pd.DataFrame(f_value, columns=["f_value"])
        f_value["features"] = self.config["features_to_change"]
        return f_value.sort_values(by="f_value", ascending=False)