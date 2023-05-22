from joblib import Parallel, delayed
from functools import partial
import logging
from tqdm import tqdm
from FastCG.utils import SmartCondition
from FastCG.utils.logger import Logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from IPython.display import display



class GeneratorBase():

    configuration_schema = {'features_to_change': {'type': list, 'schema': {'type': str}, 'required': True, 'minlength': 1},
                            'increase_only': {'type': list, 'schema': {'type': str}, 'required': False},
                            'decrease_only': {'type': list, 'schema': {'type': str}, 'required': False},
                            'max_features_to_change' : {'type': int, 'required': True, 'min': 1, 'default': 2},
                            'target': {'type': str, 'required': True},
                            'ID': {'type': str, 'required': False}}
    
    def __init__(self, all_data, model, config, target, condition, verbose=0):
        """
        Base class for all generators

        Parameters
        ----------
        all_data : list
            List of all data
        model : ModelBase
            Model to be used to generate data
        config : dict
            Configuration for the generator
        verbose : int
            Verbosity level, 0 means only show warning, 1 means show info, 2 means show debug
        """
        self.all_data = all_data.copy()
        if not self._is_valid_config(config):
            raise ValueError("Invalid config")
        self.config = config
        self.verbose = verbose
        self.model = model

        Logger.set_verbosity(verbose)

        self.target = target
        self.condition = condition
        if target is not None and condition is not None:
            self.smart_condition = SmartCondition(target, condition)

        self._preprocess_all_data()


    def update_config(self,config, preprocess=True):
        """
        Update config

        Parameters
        ----------
        config : dict
            Configuration for the generator

        Returns
        -------
        None
        """
        if not self._is_valid_config(config):
            raise ValueError("Invalid config")
        self.config = config
        if preprocess:
            self._preprocess_all_data()
             
    def _is_valid_config(self,config):
        """
        Validate config
        """
        # validate config based on schema
        try:
            for key in self.configuration_schema:
                if self.configuration_schema[key]['required'] and key not in config:
                    Logger.error(f"Missing required key {key}")
                    return False
                if key in config and not isinstance(config[key], self.configuration_schema[key]['type']):
                    Logger.error(f"Invalid type for key {key}, it is {type(config[key])} but should be {self.configuration_schema[key]['type']}")
                    return False
                if key in config and self.configuration_schema[key]['type'] == list and 'schema' in self.configuration_schema[key]:
                    for item in config[key]:
                        if not isinstance(item, self.configuration_schema[key]['schema']['type']):
                            Logger.error(f"Invalid type for key {key}, it is {type(item)} but should be {self.configuration_schema[key]['schema']['type']}")
                            return False
        except:
            return False
        return True

    def _preprocess_all_data(self):
        """
        Preprocess that runs on init, assume you have all the variables of init assigned and you can use them
        """
        pass
    
    def _preprocess_generate(self, data) -> list:
        """
        Preprocess data before generating data

        Parameters
        ----------
        data : list
            List of data to be generated

        Returns
        data : list
            List of preprocessed data
        -------
        None
        """
        Logger.debug("Preprocessing data before generating counterfactuals")
        return data
    
    def _preprocess_counterfactual_targets(self, obs) -> pd.DataFrame:
        return obs
    
    def _postprocess_valid_data(self, data) -> list:
        """
        Postprocess valid data

        Parameters
        ----------
        data : list
            List of valid data generated

        Returns
        -------
        data : list
            List of postprocessed data
        """

        Logger.debug("Postprocessing valid data")
        return data 
    
    def _postprocess_invalid_data(self, data) -> list:
        """
        Postprocess invalid data

        Parameters
        ----------
        data : list
            List of invalid data generated

        Returns
        -------
        data : list
            List of postprocessed data
        """
        Logger.debug("Postprocessing invalid data")
        return data

    def _check_chunk_for_counterfactual_targets(self, chunk):
        """
        Check if the chunk contains any counterfactual targets
        
        Parameters
        ----------
        chunk : list
            List of data to be checked
            
        Returns
        -------
        chunk : list
            List of data that contains counterfactual targets
            """
        data_to_generate = []
        for index, obs in chunk.iterrows():
            pred_result = self.model.predict(obs.to_frame().T.drop(self.id, axis=1))
            if not self.smart_condition(pred_result):
                preprocessed_obs = self._preprocess_counterfactual_targets(obs)
                data_to_generate.append((obs,preprocessed_obs))
        return data_to_generate

    
    def generate(self, data, n_jobs=-1, chunk_size=1000) -> tuple:
        """
        Generate data based on the condition

        Parameters
        ----------
        data : list
            List of data to be generated
        target : any
            Target value to be compared with
        condition : str
            Condition to be used to compare with target
        n_jobs : int
            Number of jobs to be used to generate data, -1 means using all available cores
            and 1 means using single thread
        chunk_size : int
            Size of each chunk of data to be used to generate data

        Returns
        -------
        valid_data : list
            List of valid data generated
        invalid_data : list
            List of invalid data generated
        """

        Logger.debug("Generating data")

        self.processed_data_to_generate = self._preprocess_generate(data)



        # find all data that needs to be generated
        Logger.debug("Finding all data that needs to be generated")

        
        # WIP - parallelize the finding of counterfactual targets
        # split the data into chunks and check if the chunk contains any counterfactual targets
        Logger.debug("Splitting data into chunks and checking if the chunk contains any counterfactual targets")

        chunks = []
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            chunks.append(chunk)

        if n_jobs != 1:
            data_to_generate = Parallel(n_jobs=n_jobs)(tqdm([delayed(self._check_chunk_for_counterfactual_targets)(chunk) for chunk in chunks], desc="Checking chunks for counterfactual targets in parallel"))

            # flatten list of lists and remove empty lists
            data_to_generate = [item for sublist in data_to_generate for item in sublist]
            data_to_generate = [item for item in data_to_generate if item is not None]

        else:
            data_to_generate = []
            for chunk in tqdm(chunks, desc="Checking chunks for counterfactual targets in sequencial"):
                data_to_generate += self._check_chunk_for_counterfactual_targets(chunk)
            data_to_generate = [item for item in data_to_generate if item is not None]

       

        Logger.info(f"Total data to generate: {len(data_to_generate)}")

        generated_data = []
        if n_jobs == 1:
            Logger.debug("Generating data using single thread")
            generated_data = self._generate(data_to_generate, chunk_size)
        else:
            Logger.debug("Generating data using parallel threads")
            generated_data = self._generate_parallel(
                data_to_generate, n_jobs, chunk_size)

        Logger.info(f"Total generated data: {len(generated_data)}")
        Logger.debug("Doing sanity check on generated data")

        valid_data = []
        invalid_data = []

        for obs in tqdm(generated_data, desc="Sanity check on generated data"):
            #if obs is df 
            if isinstance(obs, pd.DataFrame):
                res = self.model.predict(obs.drop(self.id, axis=1))
            else:
                obs = obs.to_frame().T.drop(self.id, axis=1)
                res = self.model.predict(obs)
            if not self.smart_condition.check(res):
                invalid_data.append(obs)
                continue
            valid_data.append(obs)

        initial_valid_data_len = len(valid_data)
        initial_invalid_data_len = len(invalid_data)
        Logger.debug(f"Total valid data: {len(valid_data)}")
        Logger.debug(f"Postprocessing valid data")

        Logger.debug("Splitting data into chunks and checking if the chunk contains any counterfactual targets")

        chunks = []
        for i in range(0, len(valid_data), chunk_size):
            chunk = valid_data[i:i + chunk_size]
            chunks.append(chunk)

        if n_jobs != 1:
            valid_data = Parallel(n_jobs=n_jobs)(tqdm([delayed(self._postprocess_valid_data)(chunk) for chunk in chunks], desc="Postprocessing valid data in parallel"))

            # merge list of dicts into a single dict containing all keys and values
            valid_data = {k: v for d in valid_data for k, v in d.items()}
        else:
            valid_data = []
            for chunk in tqdm(chunks, desc="Postprocessing valid data in sequencial"):
                valid_data.append(self._postprocess_valid_data(chunk))
            valid_data = {k: v for d in valid_data for k, v in d.items()}

        # valid_data = self._postprocess_valid_data(valid_data)
        Logger.debug(f"Total valid data after postprocessing: {len(valid_data)}")
        if len(valid_data) != initial_valid_data_len:
            Logger.warning(f"Postprocessing valid data removed {initial_valid_data_len - len(valid_data)} data points")
            
        Logger.debug(f"Total invalid data: {len(invalid_data)}")
        Logger.debug(f"Postprocessing invalid data")
        invalid_data = self._postprocess_invalid_data(invalid_data)
        Logger.debug(f"Total invalid data after postprocessing: {len(invalid_data)}")
        if len(invalid_data) != initial_invalid_data_len:
            Logger.warning(f"Postprocessing invalid data removed {initial_invalid_data_len - len(invalid_data)} data points")

        return valid_data, invalid_data

    def _generate(self, data, chunk_size):
        """
        Generate data based on the condition

        Parameters
        ----------
        data : list
            List of data to be generated
        condition : SmartCondition
            Condition to be used to compare with target
        chunk_size : int
            Size of each chunk of data to be used to generate data

        Returns
        -------
        generated_data : list
            List of generated data

        """

        # split data into chunks of chunk_size, accomidate for the last chunk can be smaller than chunk_size
        Logger.debug(
            "Splitting data into chunks of size {}".format(chunk_size))
        chunks = [data[i:i + chunk_size]
                  for i in range(0, len(data), chunk_size)]

        # generate data for each chunk
        Logger.debug(
            f"Generating data for each chunk, total chunks: {len(chunks)}")
        generated_data = []
        for chunk in tqdm(chunks, desc="Generating Counterfactuals in chunks in sequencial"):
            generated_data.extend(self._generate_chunk(chunk))

        if not generated_data:
            Logger.critical("No data generated")
            raise ValueError("No data generated")
        
        return generated_data

    def _generate_parallel(self, data, n_jobs, chunk_size):
        """
        Generate data based on the condition using parallel threads

        Parameters
        ----------
        data : list
            List of data to be generated
        condition : SmartCondition
            Condition to be used to compare with target
        n_jobs : int
            Number of jobs to be used to generate data, -1 means using all available cores
            and 1 means using single thread
        chunk_size : int
            Size of each chunk of data to be used to generate data

        Returns
        -------
        generated_data : list
            List of generated data

        """
        # split data into chunks of chunk_size, accomidate for the last chunk can be smaller than chunk_size
        Logger.debug(
            "Splitting data into chunks of size {}".format(chunk_size))
        chunks = [data[i:i + chunk_size]
                  for i in range(0, len(data), chunk_size)]

        # generate data for each chunk
        Logger.debug(
            f"Generating data for each chunk, total chunks: {len(chunks)}")
        generated_data = Parallel(n_jobs=n_jobs, require='sharedmem')(
            delayed(self._generate_chunk)(chunk) for chunk in tqdm(chunks, desc="Generating Counterfactual in chunks in parallel"))

        if not generated_data:
            Logger.critical("No data generated")
            raise ValueError("No data generated")

        # flatten the generated data in case of list of lists
        if isinstance(generated_data[0], list):
            generated_data = [
                item for sublist in generated_data for item in sublist]

        return generated_data

    def _generate_chunk(self, chunk):
        # logging.debug(f"Generating data for chunk of size {len(chunk)}")
        all_generated_data = []
        for obs,processed_obs in chunk:
            generated_data = self._generate_single(obs,processed_obs)
            if generated_data is not None and len(generated_data) > 0:
                all_generated_data.append(generated_data)
        return all_generated_data

    def _generate_single(self, obs,processed_obs):
        raise NotImplementedError
    
    def show_counterfactual(self, key, counterfactual_list, show_plot=False):
        """
        Show the counterfactual for a given ID.

        Parameters
        ----------
        key : int
            The ID.
        
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
        self.obs = self.all_data[self.all_data[self.id] == key].copy().drop(self.config["target"], axis=1)
        for col in self.all_data.columns:
            if col != self.config["target"]:
                results.loc[0, col] = self.obs[col].values[0].astype(int)
                results.loc[1, col] = counterfactual[col].values[0].astype(int)
        results = results.T
        results.columns = ["Original", "Counterfactual"]
        self.counterfactual = results["Counterfactual"].to_frame().T
        results["Difference"] = results["Counterfactual"] - results["Original"]
        results["% Difference"] = results["Difference"] / results["Original"] * 100 
        results["% Difference"] = results["% Difference"].fillna(0)
        results["% Difference"] = results["% Difference"].replace([np.inf, -np.inf], 0)
        # print("In order to change the prediction from ", self.obs[self.config["target"]].values[0], " to ", counterfactual[self.config["target"]].values[0], " do the following:")
        print("-------------------------------------------------------------------")
        print("ID: ", key)
        print("-------------------------------------------------------------------")
        for row in results.iterrows():
            if row[0] == self.id:
                    continue
            if results["Difference"][row[0]] > 0:
                print("Increase ", row[0], " by ", row[1]["% Difference"], "%") 
                print("Increase ", row[0], " by ", row[1]["Difference"], " units")
            elif results["Difference"][row[0]] < 0:
                print("Decrease ", row[0], " by ", row[1]["% Difference"], "%")
                print("Decrease ", row[0], " by ", row[1]["Difference"], " units")

        if show_plot:
            self.plot_the_counterfactual(self.obs, self.counterfactual)

        display(results)
        return self.obs, self.counterfactual

    def get_counterfactual(self, key, counterfactual_list):
        """
        Get the counterfactual for a given ID. 

        Parameters
        ----------
        key : int
            The ID.
        
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
        obs = self.all_data[self.all_data[self.id] == key].copy().drop(self.config['target'], axis=1)
        for col in self.all_data.columns:
            if col != self.config["target"]:
                results.loc[0, col] = self.obs[col].values[0]
                results.loc[1, col] = counterfactual[col].values[0]
        results = results.T
        results.columns = ["Original", "Counterfactual"]
        counterfactual = results["Counterfactual"].to_frame().T
        return obs, counterfactual
    

    def plot_the_counterfactual(self, obs, counterfactual):
        """
        Plot the counterfactual for a given ID.

        Parameters
        ----------
        obs : pd.DataFrame
            The original observation.

        counterfactual : pd.DataFrame
            The counterfactual.

        Returns
        -------
        None
        
        """
        if self.config["target"] in obs.columns:
            df2 = obs.drop(self.config["target"], axis=1).reset_index(drop=True)
        else:
            df2 = obs.copy().reset_index(drop=True)


        # mock df with 3 columns 
        df = counterfactual.reset_index(drop=True)
        #turn the df to int values
        df = df.astype(int)
        df2 = df2.astype(int)

        print(df2.shape)
        print(counterfactual.shape)
        # display(obs)
        # display(counterfactual)
        # diff df 
        df3 = df - df2


        # max of df and df2
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
