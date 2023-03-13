from sklearn.neighbors import KNeighborsClassifier 
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial

class CounterfactualGenerator(object):
    def __init__(self, all_data, model, config, target_class):
        self.model = model
        self.config = config
        self.target_class = target_class
        self.all_data = all_data
    
    def generate_single_counterfactual(self, obs):
        raise NotImplementedError 

    def pre_process(self):
        pass 
    
    def generate_mulitple_counterfactuals(self, data):
        counterfactual_list = []
        if data.empty:
            return counterfactual_list
        for _ , obs in data.iterrows():
            if self.model.predict(obs.to_frame().T)[0] != self.target_class:
                counterfactual = self.generate_single_counterfactual(obs)
                if counterfactual is not None:
                    counterfactual_list.append(counterfactual)
        return counterfactual_list

    def generate_mulitple_counterfactuals_parallel(self, data, segment_size=1000,n_jobs=-1):
        jobs = []
        self.pre_process()
        # split the data into segments with the size of segment_size
        for i in range(0, len(data), segment_size):
            jobs.append(delayed(self.generate_mulitple_counterfactuals)(data.iloc[i:i+segment_size]))
        
        # run the jobs in parallel
        counterfactual_list = Parallel(n_jobs=n_jobs)(jobs)

        # flatten the list of lists
        counterfactual_list = [item for sublist in counterfactual_list for item in sublist]

        return counterfactual_list


class CFKnn(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        #config is a dictionary
        self.all_data = all_data
        self.config = config 
        self.model = model
        self.target_class = target_class
        

    def pre_process(self):
        self.nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(self.all_data[self.config["features_to_change"]])

    def generate_mulitple_counterfactuals(self, data):
        counterfactual_list = [] 
        for _ , obs in data.iterrows():
            if self.model.predict(obs.to_frame().T)[0] == self.target_class:
                continue
            else:
                counterfactual = self._generate_single_counterfactual(obs, self.nbrs)
                if counterfactual is not None:
                    counterfactual_list.append(counterfactual)
        return counterfactual_list
    
    def _generate_single_counterfactual(self, obs, nbrs):
        _ , indices = nbrs.kneighbors(obs[self.config["features_to_change"]].to_frame().T)
        for i in indices[0]:
            counterfactual = obs.copy()
            for feature in self.config["features_to_change"]:
                counterfactual[feature] = self.all_data.iloc[i][feature]
                if self.model.predict(counterfactual.to_frame().T)[0] == self.target_class: 
                    return counterfactual
        return None
            


class InstanceBasedCF(CounterfactualGenerator):
    def __init__(self, model, config, target_class):
        self.config = config
        self.model = model
        self.target_class = target_class
 
    # def generate_mulitple_counterfactuals(self, data):
    #     counterfactual_list = []
    #     if data.empty:
    #         return counterfactual_list
    #     for _ , obs in data.iterrows():
    #         if self.model.predict(obs.to_frame().T)[0] != self.target_class:
    #             counterfactual = self._find_counterfactoral(self.config["features_to_change"], self.target_class, obs)
    #             if counterfactual is not None:
    #                 counterfactual_list.append(counterfactual)
    #     return counterfactual_list

    def generate_single_counterfactual(self, obs):
        for i in np.arange(0.8, 5.0, 0.1):
            counterfactual = obs.copy() 
            for feature in self.config["features_to_change"]:
                counterfactual[feature] = self._change_feature_by_percent(obs, feature, i)
                if self.model.predict(counterfactual.to_frame().T)[0] == self.target_class:
                    return counterfactual

    def _change_feature_by_percent(self, obs, feature, precent): 
        obs = obs.copy()
        obs[feature] = obs[feature] * precent
        return obs[feature]
        
        