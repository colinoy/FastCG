from FCG.CounterfactualGenerator import CounterfactualGenerator
import numpy as np


class InstanceBasedCF(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class)
 
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

