from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.neighbors import KNeighborsClassifier
from functools import partial
from scipy.spatial.distance import minkowski
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class FreezeKNNCF(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class)
        self.nbrs = None


    def pre_process(self):
        weights = []
        self.data_class = self.all_data[self.model.predict(self.all_data.drop(self.config["target"], axis=1)) == self.target_class]
        for feature in self.all_data.drop(self.config["target"], axis=1).columns:
            if feature == "Loan ID" or feature == "Customer ID":
                weights.append(0)
            elif feature not in self.config["features_to_change"]:
                weights.append(1)
            else:
                weights.append(0)
        
        w_minkowski = partial(minkowski, p=2, w=weights) #weighted minkowski distance 
        self.nbrs = KNeighborsClassifier(n_neighbors=20, metric=w_minkowski, algorithm="ball_tree", n_jobs=-1).fit(self.data_class.drop(self.config["target"], axis=1), self.data_class[self.config["target"]])
    
    def generate_single_counterfactual(self, obs):
        print("Getting into generate_single_counterfactual in FreezeKNNCF.py")
        #generate one counterfactual for one instance 
        counterfactual = obs.copy()
        _, indices = self.nbrs.kneighbors(obs.to_frame().T)
        for indice in indices[0]:
            for feature in self.config["features_to_change"]:
                counterfactual[feature] = self.data_class.iloc[indice][feature]
                if self.model.predict(counterfactual.to_frame().T)[0] == self.target_class:
                    return counterfactual
        return None 


    def generate_mulitple_counterfactuals(self, data_to_genrate):
        print("Getting into generate_mulitple_counterfactuals in FreezeKNNCF.py")
        #creat knn model using all the data and all the features not to be changed
        counterfactual_list = {}
        for _, obs in data_to_genrate.iterrows():
            obs = obs.drop(self.config["target"])
            if self.model.predict(obs.to_frame().T)[0] == self.target_class:
                continue
            print("Found an instance to generate counterfactual for")
            counterfactual = self.generate_single_counterfactual(obs)
            if counterfactual is not None:
                counterfactual = self.improve_with_binary_search_by_vectors(obs, counterfactual)
                counterfactual_list[obs["Loan ID"].astype(int)] = counterfactual
        return counterfactual_list
    

    def calculate_point(self,start,direction,step):
        return start + direction*step

    def calculate_distance(self,point1,point2):
        return np.linalg.norm(point1-point2)

    def calculate_direction(self,point1,point2):
        return (point2-point1)/np.linalg.norm(point2-point1)

    def improve_with_binary_search_by_vectors(self,start_point,end_point):
        distance = self.calculate_distance(start_point,end_point)
        direction = self.calculate_direction(start_point,end_point)

        start_distance = 0
        end_distance = distance

        while abs(start_distance - end_distance) < 0.0001:
            mid_point = self.calculate_point(start_point,direction,(start_distance+end_distance)/2)
            if self.model.predict(mid_point) == self.target_class:
                end_distance = (start_distance+end_distance)/2
            else:
                start_distance = (start_distance+end_distance)/2

        return self.calculate_point(start_point,direction,(start_distance+end_distance)/2)
        