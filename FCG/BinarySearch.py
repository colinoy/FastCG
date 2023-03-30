from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class BinarySearch(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class)
        self.normalizer = StandardScaler().fit(self.all_data[self.config["features_to_change"]])    
        self.n_components = 1
        self.pca = PCA(n_components=self.n_components).fit(self.normalizer.transform(self.all_data[self.config["features_to_change"]]))
        self.features_pca = ['principal component ' + str(i) for i in range(1, self.n_components+1)]


    def pre_process(self):
        pass 
    

    def generate_single_counterfactual(self, obs, data_to_genrate_copy):
        obs = pd.DataFrame(obs).T
        starting_point = obs[self.features_pca] * 0
        end_point = obs[self.features_pca] * 2
        #take the value from the numpy array and convert it to a float 
        starting_point = starting_point.values[0]
        end_point = end_point.values[0]
        while (end_point - starting_point).sum() > 0.0001:
            mid_point = (end_point + starting_point) * 0.5
            obs[self.features_pca] = mid_point
            obs_not_pca = self.convert_obs_before_PCA(obs, data_to_genrate_copy)
            if self.model.predict(obs_not_pca) == self.target_class:
                end_point = mid_point
            else:
                starting_point = mid_point

        obs[self.features_pca] = end_point
        obs = self.convert_obs_before_PCA(obs, data_to_genrate_copy)
        if self.model.predict(obs) == self.target_class:
            return obs
        else:
            return None

    def generate_mulitple_counterfactuals(self, data_to_genrate): 
        counterfactual_list = {}
        print("Getting into generate_mulitple_counterfactuals in BinarySearch.py")
        if "Loan Status" in data_to_genrate.columns:
            data_to_genrate = data_to_genrate.drop("Loan Status", axis=1)
        data_to_genrate = data_to_genrate[self.model.predict(data_to_genrate) != self.target_class]
        data_to_genrate_copy = data_to_genrate.copy()
        data_to_genrate = self.convert_obs_to_pca(data_to_genrate)
        for _, obs in data_to_genrate.iterrows():
            counterfactual = self.generate_single_counterfactual(obs.copy(), data_to_genrate_copy)
            if counterfactual is not None:
                counterfactual_list[obs["Loan ID"].astype(int)] = counterfactual
        return counterfactual_list 
    

    def precent_change(self, obs, counterfactual):
        return (counterfactual - obs) / obs * 100
    

    def convert_obs_before_PCA(self, obs, data_to_genrate_copy):
        obs_copy = obs.copy()
        obs = obs[self.features_pca]
        obs = self.pca.inverse_transform(obs)
        obs = self.normalizer.inverse_transform(obs)
        obs = pd.DataFrame(obs, columns=self.config["features_to_change"])
        obs = obs.reset_index(drop=True)
        obs_copy = obs_copy.reset_index(drop=True)
        obs = pd.concat([obs, obs_copy.drop(self.features_pca, axis=1)], axis=1)
        obs = obs[self.all_data.drop("Loan Status", axis=1).columns]
        #convert the features to change to int 
        obs[self.config["features_to_change"]] = obs[self.config["features_to_change"]].astype(int)
        return obs
    

    def convert_obs_to_pca(self, obs):
        obs_copy = obs.copy()
        obs = self.normalizer.transform(obs[self.config["features_to_change"]])
        obs = self.pca.transform(obs)
        obs = pd.DataFrame(obs, columns=self.features_pca)
        obs = obs.reset_index(drop=True)
        obs_copy = obs_copy.reset_index(drop=True)
        obs = pd.concat([obs, obs_copy.drop(self.config["features_to_change"], axis=1)], axis=1)
        return obs
    