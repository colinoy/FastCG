import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial
import traceback


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
        print("Getting into generate_mulitple_counterfactuals_parallel in CounterfactualGenerator.py")
        jobs = []
        self.pre_process()

        # run the jobs in parallel unless n_jobs is set to 1 == sequential
        if n_jobs == 1:
            print("Running in sequential mode")
            counterfactual_list = []

            for i in range(0, len(data), segment_size):
                segment = data.iloc[i:i+segment_size]
                try:
                    counterfactual_list.append(self.generate_mulitple_counterfactuals(segment))
                except Exception as e:
                    print(traceback.format_exc())
                    
        else:
            for i in range(0, len(data), segment_size):
                segment = data.iloc[i:i+segment_size]
                jobs.append(delayed(self.generate_mulitple_counterfactuals)(segment))
            
            counterfactual_list = Parallel(n_jobs=n_jobs)(jobs)
        
        print("Done generating counterfactuals")

        # flatten the list of lists
        counterfactual_list = {k: v for d in counterfactual_list for k, v in d.items()}

        return counterfactual_list
    
    def show_counterfactoal(self, key, counterfactual_list):
        results = pd.DataFrame()
        counterfactual = counterfactual_list.get(key)
        obs = self.all_data[self.all_data["Loan ID"] == key]
        for col in self.all_data.columns:
            if col != "Loan Status":
                results.loc[1, col] = obs[col].values[0]
                results.loc[2, col] = counterfactual[col]
        results  = results.T
        results.columns = ["Original", "Counterfactual"]
        results["Difference"] = results["Counterfactual"] - results["Original"]
        results["% Difference"] = results["Difference"] / results["Original"] * 100 
        results["% Difference"] = results["% Difference"].fillna(0)
        print("In order for the loan to be approved, the following changes need to be made:")
        print("-------------------------------------------------------------------")
        print("Loan ID: ", key)
        print("-------------------------------------------------------------------")
        for row in results.iterrows():
            if results["Difference"][row[0]] > 0:
                print("Increase ", row[0], " by ", row[1]["% Difference"], "%") 
            elif results["Difference"][row[0]] < 0:
                print("Decrease ", row[0], " by ", row[1]["% Difference"], "%")

        display(results)

       


