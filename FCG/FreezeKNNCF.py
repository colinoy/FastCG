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
        self.normalizer = StandardScaler().fit(self.all_data[self.config["features_to_change"]])    
        self.n_components = self.choose_pca_components()
        self.pca = PCA(n_components=self.n_components).fit(self.normalizer.transform(self.all_data[self.config["features_to_change"]]))
        self.features_pca = ['principal component ' + str(i) for i in range(1, self.n_components+1)]


    def pre_process(self):
        weights = []
        self.data_class = self.all_data[self.model.predict(self.all_data.drop(self.config["target"], axis=1)) == self.target_class]
        self.data_pca = self.data_class[self.config["features_to_change"]]
        self.data_pca = self.normalizer.transform(self.data_pca)
        self.data_pca = self.pca.transform(self.data_pca)
        self.data_pca = pd.DataFrame(data = self.data_pca, columns = self.features_pca)
        self.data_pca.reset_index(drop=True, inplace=True)
        self.data_class.reset_index(drop=True, inplace=True)
        self.data_pca = pd.concat([self.data_pca, self.data_class.drop(self.config["features_to_change"], axis=1)], axis=1)
        for feature in self.data_pca.drop(self.config["target"], axis=1).columns:
            if feature == "Loan ID" or feature == "Customer ID":
                weights.append(0) 
            elif feature not in self.features_pca:
                weights.append(1)
            else:
                weights.append(0)
        w_minkowski = partial(minkowski, p=2, w=weights) #weighted minkowski distance 
        self.nbrs = KNeighborsClassifier(n_neighbors=5, metric=w_minkowski, algorithm="ball_tree", n_jobs=-1).fit(self.data_pca.drop(self.config["target"], axis=1), self.data_pca[self.config["target"]])
    
    def generate_single_counterfactual(self, obs):
        #generate one counterfactual for one instance 
        _, indices = self.nbrs.kneighbors(obs)
        for indice in indices[0]:
            count_index = self.data_class["Loan ID"].iloc[indice] 
            counterfactual = self.data_pca.iloc[indice][self.features_pca]
            counterfactual_pred = self.pca.inverse_transform(counterfactual).reshape(1, -1)
            counterfactual_pred = pd.DataFrame(data = counterfactual_pred, columns = self.config["features_to_change"])
            counterfactual_pred = self.normalizer.inverse_transform(counterfactual_pred)
            counterfactual_pred = pd.DataFrame(data = counterfactual_pred, columns = self.config["features_to_change"])
            obs.reset_index(drop=True, inplace=True)
            counterfactual_pred.reset_index(drop=True, inplace=True) 
            counterfactual_pred = pd.concat([counterfactual_pred, obs.drop(self.features_pca, axis=1)], axis=1)
            #sort the columns to match the original data 
            counterfactual_pred = counterfactual_pred[self.all_data.drop(self.config["target"], axis=1).columns]
            #check if the counterfactual is result with the desired class 
            if self.model.predict(counterfactual_pred) == self.target_class:
                #convert the counterfactual to int 
                counterfactual_pred[self.config["features_to_change"]] = counterfactual_pred[self.config["features_to_change"]].astype(int)
                counterfactual_pred["Loan ID"] = count_index
                return counterfactual_pred
        return None 


    def generate_mulitple_counterfactuals(self, data_to_genrate):
        counterfactual_list = {}
        data_to_genrate = data_to_genrate[self.model.predict(data_to_genrate.drop(self.config["target"], axis=1)) != self.target_class]
        data_to_genrate_norm = self.normalizer.transform(data_to_genrate[self.config["features_to_change"]])
        data_to_genrate_norm = self.pca.transform(data_to_genrate_norm)
        data_to_genrate_norm = pd.DataFrame(data = data_to_genrate_norm, columns = self.features_pca)
        data_to_genrate_norm = data_to_genrate_norm.reset_index(drop=True)
        data_to_genrate = data_to_genrate.reset_index(drop=True)
        data_to_genrate = pd.concat([data_to_genrate_norm, data_to_genrate.drop(self.config["features_to_change"], axis=1)], axis=1)
        for _, obs in data_to_genrate.iterrows():
            obs = obs.to_frame().T.drop(self.config["target"], axis=1)
            counterfactual = self.generate_single_counterfactual(obs)
            if counterfactual is not None:
                new_obs = obs.copy() 
                new_obs = self.pca.inverse_transform(new_obs[self.features_pca]).reshape(1, -1)
                new_obs = pd.DataFrame(data = new_obs, columns = self.config["features_to_change"]) 
                new_obs = self.normalizer.inverse_transform(new_obs)
                new_obs = pd.DataFrame(data = new_obs, columns = self.config["features_to_change"])
                new_obs = pd.concat([new_obs, obs.drop(self.features_pca, axis=1)], axis=1)
                counterfactual = self.improve_with_binary_search_by_vectors(new_obs, counterfactual)
                counterfactual_list[obs["Loan ID"].values[0]] = counterfactual
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
    

    def choose_pca_components(self):
        pca = PCA(n_components=0.8)
        pca.fit(self.normalizer.transform(self.all_data[self.config["features_to_change"]]))
        return pca.n_components_ 
        
        