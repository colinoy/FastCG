import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from functools import partial
import traceback
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class CounterfactualGenerator(object):
    def __init__(self, all_data, model, config, target_class) -> None:
        """
        Initialize the counterfactual generator object with the model, config and target class. 

        Parameters
        ----------
        all_data : pd.DataFrame
            The data used to generate counterfactuals.

        model : sklearn model
            The model used to generate counterfactuals.

        config : dict
            The configuration used to generate counterfactuals. Includes the following keys:
            - "features_to_change": list of features to change 
            - "target": the target feature
        
        target_class : int
            - 0: Loan is not approved
            - 1: Loan is approved 

        Returns
        -------
        None
        
        """
        self.model = model
        self.config = config
        self.target_class = target_class
        self.all_data = all_data
    
    def generate_single_counterfactual(self, obs):
        raise NotImplementedError 

    def pre_process(self):
        pass 
    
    def generate_mulitple_counterfactuals(self, data):
        """
        Generate multiple counterfactuals for a given dataset.

        Parameters
        ----------
        data : pd.DataFrame
            The data used to generate counterfactuals.
        
        Returns
        -------
        counterfactual_list : list
            A list of counterfactuals.
        
        """
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
        """

        Run the counterfactual generation in parallel using joblib in order to speed up the process. 

        Parameters
        ----------
        data : pd.DataFrame
            The data used to generate counterfactuals.
        
        segment_size : int
            The size of the segment to be processed in parallel. Default is 1000.

        n_jobs : int
            The number of jobs to run in parallel. Default is -1 which means all available cores will be used.

        Returns
        -------
        counterfactual_list : list
            A list of counterfactuals.
        
        """


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
    
    def show_counterfactoal(self, key, counterfactual_list, show_plot=False):
        """
        Show the counterfactual for a given loan ID.

        Parameters
        ----------
        key : int
            The loan ID.
        
        counterfactual_list : list
            A list of counterfactuals.

        show_plot : bool
            Whether to show the plot or not. Default is False.

        Returns
        -------
        None
    
        """
        results = pd.DataFrame()
        counterfactual = counterfactual_list.get(key)
        self.obs = self.all_data[self.all_data["Loan ID"] == key].copy().drop("Loan Status", axis=1)
        for col in self.all_data.columns:
            if col != "Loan Status":
                results.loc[0, col] = self.obs[col].values[0].astype(int)
                results.loc[1, col] = counterfactual[col].values[0].astype(int)
        results = results.T
        results.columns = ["Original", "Counterfactual"]
        self.counterfactual = results["Counterfactual"].to_frame().T
        results["Difference"] = results["Counterfactual"] - results["Original"]
        results["% Difference"] = results["Difference"] / results["Original"] * 100 
        results["% Difference"] = results["% Difference"].fillna(0)
        results["% Difference"] = results["% Difference"].replace([np.inf, -np.inf], 0)
        print("In order for the loan to be approved, the following changes need to be made:")
        print("-------------------------------------------------------------------")
        print("Loan ID: ", key)
        print("-------------------------------------------------------------------")
        for row in results.iterrows():
            if row[0] == "Loan ID":
                    continue
            if results["Difference"][row[0]] > 0:
                print("Increase ", row[0], " by ", row[1]["% Difference"], "%") 
            elif results["Difference"][row[0]] < 0:
                print("Decrease ", row[0], " by ", row[1]["% Difference"], "%")

        display(results)

        if show_plot:
            self.plot_the_counterfactoal(self.obs, self.counterfactual)

        return self.obs, self.counterfactual

    def get_counterfactual(self, key, counterfactual_list):
        """
        Get the counterfactual for a given loan ID. 

        Parameters
        ----------
        key : int
            The loan ID.
        
        counterfactual_list : list
            A list of counterfactuals.

        Returns
        -------
        obs : pd.DataFrame
            The original observation.
        
        counterfactual : pd.DataFrame
            The counterfactual.
        
        """

        results = pd.DataFrame()
        counterfactual = counterfactual_list.get(key)
        obs = self.all_data[self.all_data["Loan ID"] == key].copy().drop("Loan Status", axis=1)
        for col in self.all_data.columns:
            if col != "Loan Status":
                results.loc[0, col] = self.obs[col].values[0]
                results.loc[1, col] = counterfactual[col].values[0]
        results = results.T
        results.columns = ["Original", "Counterfactual"]
        counterfactual = results["Counterfactual"].to_frame().T
        return obs, counterfactual
    

    def plot_the_counterfactoal(self, obs, counterfactoal):
        """
        Plot the counterfactual for a given loan ID.

        Parameters
        ----------
        obs : pd.DataFrame
            The original observation.

        counterfactoal : pd.DataFrame
            The counterfactual.

        Returns
        -------
        None
        
        """
        if self.config["target"] in obs.columns:
            df = obs.drop(self.config["target"], axis=1)
        else:
            df = obs.copy()

        # mock df with 3 columns 
        df2 = counterfactoal.reset_index(drop=True)
        # display(obs)
        # display(counterfactoal)
        # diff df 
        df3 = df - df2

        # max of df and df2
        display(df)
        display(df2)
        df_max = df.where(df > df2, df2, axis=0)

        # min of df and df2
        df_min = df.where(df < df2, df2, axis=0)

        # df_red contains max of df and df2 if df3 <0 else 0
        df_red = df_max.where(df3 < 0, 0, axis=0)

        df_green = df_max.where(df3 > 0, 0, axis=0)

        # df_gray contains the minimum value between df and df2
        df_gray = df_min

        # plot the data, overlay the plots in the following order green,red and then gray
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.xticks(rotation=90)
        # green bar is only outline of the bar

        width = 0.5
        ax.bar(df_green.columns, df_green.iloc[0], hatch='//', linewidth=2, edgecolor='green', fill=False, width=width)
        ax.bar(df_red.columns, df_red.iloc[0], color='gray',linewidth=2, edgecolor='red', hatch='//', width=width,  )
        ax.bar(df_gray.columns, df_gray.iloc[0], color='gray',linewidth=2, edgecolor='gray', hatch='', width=width)

        # for each bar in the plot add a text label on the top of the bar containing the difference between df and df2
        patches = {}
        # group pathces by column
        for i,p in enumerate(ax.patches):
            col = p.get_x()
            if col not in patches:
                patches[col] = []
            patches[col].append(p)


        for index,col_x in enumerate(patches):


            label_data = df3.iloc[0, index].round(2)
            if label_data > 0:
                label = u'$\u25B2$' + f'{label_data}'
            elif label_data < 0:
                label = u'$\u25BC$' + f'{label_data}'
            else:
                label = ''

            
            max_height = max([p.get_height() for p in patches[col_x]])
            width = patches[col_x][0].get_width() / 2
            # add the text label
            ax.annotate(label,
                        (col_x + width, max_height),
                        ha='center', va='center', 
                        fontsize=11, color='green' if label_data > 0 else 'red', 
                        xytext=(0, 8), textcoords='offset points')
        
    




       


