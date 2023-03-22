from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.neighbors import NearestNeighbors

class CFKnn(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class)
        

    def pre_process(self):
        self.nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree').fit(self.all_data[self.config["features_to_change"]])

    def generate_mulitple_counterfactuals(self, data):
        counterfactual_list = {}
        for _ , obs in data.iterrows():
            if self.model.predict(obs.to_frame().T)[0] == self.target_class:
                continue
            else:
                counterfactual = self._generate_single_counterfactual(obs, self.nbrs)
                if counterfactual is not None:
                    counterfactual_list[obs["Loan ID"].astype(int)] = counterfactual
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
     