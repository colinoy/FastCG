from FCG.CounterfactualGenerator import CounterfactualGenerator

class BinarySearch(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class)
        print("Getting into BinarySearch.py")


    def pre_process(self):
        pass 
    


    def generate_single_counterfactual(self, obs, feature, max_precent_change=100):
        starting_point = obs[feature] * (1 - max_precent_change / 100)
        end_point = obs[feature] * (1 + max_precent_change / 100)
        # try:
        #     while self.model.predict(obs.to_frame().T)[0] != self.target_class:
        #         end_point = end_point * 2 
        #         obs[feature] = end_point
        # except:
        #     if end_point > self.all_data[feature].max():
        #         end_point = self.all_data[feature].max()
        #         obs[feature] = end_point
                # if self.model.predict(obs.to_frame().T)[0] != self.target_class:
                #     return None  
        # if self.model.predict(obs.to_frame().T)[0] != self.target_class:
        #     return None
    
        while abs(starting_point - end_point) > 10:
            mid_point = (end_point + starting_point) * 0.5
            obs[feature] = mid_point
            if self.model.predict(obs.to_frame().T)[0] == self.target_class:
                end_point = mid_point
            else:
                starting_point = mid_point
        obs[feature] = end_point
        if self.model.predict(obs.to_frame().T)[0] == self.target_class:
            return obs
        else:
            return None
            

    def generate_mulitple_counterfactuals(self, data_to_genrate): 
        counterfactual_list = {}
        print("Getting into generate_mulitple_counterfactuals in BinarySearch.py")
        for _, obs in data_to_genrate.iterrows():
            if self.config["target"] in data_to_genrate.columns:
                obs = obs.drop(self.config["target"])
            if self.model.predict(obs.to_frame().T)[0] == self.target_class:
                continue
            for i in range(len(self.config["features_to_change"])):
                counterfactual = self.generate_single_counterfactual(obs.copy(), self.config["features_to_change"][i])
                if counterfactual is not None:
                    counterfactual_list[obs["Loan ID"].astype(int)] = counterfactual
                    break
        return counterfactual_list 
    

    def precent_change(self, obs, counterfactual):
        return (counterfactual - obs) / obs * 100