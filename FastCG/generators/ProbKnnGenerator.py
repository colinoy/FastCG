import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import minkowski
from functools import partial
from IPython.display import display
from  tqdm import tqdm
from FastCG.generators import GeneratorBase
from FastCG.utils.util_functions import find_centroid, find_feature_improtance_Anova, choose_pca_components, convert_to_pca, find_feature_direction, convert_from_pca,obs_distance
from FastCG.utils.util_functions import improve_with_binary_search_by_vectors
from FastCG.utils.logger import Logger


# Logger.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%Y-%m-%d:%H:%M:%S')

class ProbKnnGenerator(GeneratorBase):

    def _preprocess_all_data(self):
        self.data_matching_condition = \
                self.all_data[
                    self.smart_condition.check(self.model.predict(self.all_data.drop([self.config["target"]], axis=1)))
                    ]
        self.features_to_use = \
            find_feature_improtance_Anova(
                self.data_matching_condition, 
                self.config["max_features_to_change"],
                self.config["features_to_change"], 
                self.config["target"]
                )
        if "increase_only" in self.config and self.config["increase_only"] is not None:
            self.features_increase_only = self.config["increase_only"]
        else: 
            self.features_increase_only = []
        if "decrease_only" in self.config and self.config["decrease_only"] is not None:
            self.features_decrease_only = self.config["decrease_only"]
        else: 
            self.features_decrease_only = []
        Logger.info("Features to use: " + str(self.features_to_use))
        Logger.info("Features to increase: " + str(self.features_increase_only))
        Logger.info("Features to decrease: " + str(self.features_decrease_only))

        self.all_data_features_to_use = self.all_data[self.features_to_use]
        self.all_data["ID"] = self.all_data.index
        self.id = "ID"
        self.normalizer = StandardScaler().fit(self.all_data_features_to_use)
        self.data_normalized_features_to_use = self.normalizer.transform(self.all_data_features_to_use)
        
        self.n_components = choose_pca_components(self.data_normalized_features_to_use, 0.75)
        self.pca = PCA(n_components=self.n_components).fit(self.data_normalized_features_to_use)
        self.features_pca = ['principal component ' + str(i) for i in range(1, self.n_components+1)]
        self.data_pca = convert_to_pca(self.data_matching_condition, self.pca, self.normalizer, self.features_to_use, self.features_pca)
        self.data_centroid, self.kmean_lables = find_centroid(self.data_pca.drop(self.config["target"], axis=1).copy()[self.features_pca],features_pca=self.features_pca)

        Logger.debug(f"training on centroids with the following features: {self.features_pca}")
        self.nbrs = NearestNeighbors(n_neighbors=2).fit(self.data_centroid) #TODO: move n_neighbors to config

    def _preprocess_generate(self, data) -> None:
        self.current_data = data
        self.current_data["ID"] = self.current_data.index
    
    def _preprocess_counterfactual_targets(self, obs) -> pd.DataFrame:
        data_to_genrate = convert_to_pca(obs.to_frame().T, self.pca, self.normalizer, self.features_to_use, self.features_pca)
        data_to_genrate = data_to_genrate.fillna(0)
        return data_to_genrate

    def __init__(self, all_data, model, config, target, condition,verbose=0):
        super().__init__(all_data, model, config, target, condition,verbose)

    def _generate_single(self, original_obs, processed_obs):
        # Logger.info("Generating counterfactual for observation: " + str(processed_obs))
        indices = self.knn_on_cluster(processed_obs)
        base_obs = self.current_data[self.current_data[self.id] == processed_obs[self.id].values[0]]
        cf = {}
        for idx in indices:
            counterfactual = processed_obs.copy()
            for feature in self.features_pca: 
                counterfactual[feature] = self.data_pca[feature].iloc[idx]
            
            counterfactual = convert_from_pca(counterfactual,base_obs, self.pca, self.normalizer, self.features_to_use, self.features_pca)
            # to_continue = False
            for feature in self.features_increase_only:
                if counterfactual[feature].values[0] < original_obs[feature]:
                    counterfactual[feature].values[0] = original_obs[feature]*2
            for feature in self.features_decrease_only:
                if counterfactual[feature].values[0] > original_obs[feature]:
                    counterfactual[feature].values[0] = original_obs[feature]/2            
            tmp_counterfactual = counterfactual.copy()
            if self.config["target"] in tmp_counterfactual:
                    tmp_counterfactual = counterfactual.drop(self.config["target"], axis=1)

            if self.smart_condition.check(self.model.predict(tmp_counterfactual.drop(self.id, axis=1)))[0]:
                # return counterfactual
                distance = obs_distance(counterfactual.copy(), base_obs.copy(), self.config["target"])
                if distance not in cf:
                    cf[distance] = counterfactual.copy()
        if cf:
            return cf[min(cf.keys())]
        else:
            return original_obs
    
    def knn_on_cluster(self,processed_obs):
        """
        
        Performs the KNN algorithm on the closest cluster to the original observation in order to find the closeset instance to the original observation in the cluster.

        Parameters
        ----------
        obs : pd.DataFrame
            The original observation.
        
        Returns
        -------
        pd.DataFrame
            The closest instance to the original observation in the cluster.
        
        """
        obs_original = processed_obs[self.data_pca.drop(self.config["target"], axis=1).columns]
        processed_obs = processed_obs[self.features_pca]
        indices = self.nbrs.kneighbors(processed_obs, return_distance=False) 
        index = []

        cluster_ids = [i for i in indices[0]]
        all_cluster_members = {i: [] for i in cluster_ids}
        for id in cluster_ids:
            all_cluster_members[id] = np.where(self.kmean_lables == id)[0]
            
        for indice in indices[0]:
            cluster_members = all_cluster_members[indice] #[j for j, x in enumerate(self.kmean_lables) if x == indice]
            max_index = min(len(cluster_members), 5)
            for member in cluster_members[:max_index]:
                idx = self.data_pca.iloc[member].name
                if idx not in index:
                    index.append(idx)
        
        weights = []
        data_cluster = self.data_pca.loc[index] 
        for feature in data_cluster.drop(self.config["target"], axis=1).columns: 
            if feature == self.id:
                weights.append(0)  
            elif feature not in self.features_pca: 
                weights.append(1)
            else:
                weights.append(0)
        w_minkowski = partial(minkowski, p=2, w=weights)
        n_neigh = min(5, len(data_cluster))
        nbrs = NearestNeighbors(n_neighbors=n_neigh, metric=w_minkowski).fit(data_cluster.drop(self.config["target"], axis=1))
        indices = nbrs.kneighbors(obs_original, return_distance=False)
        return indices[0]
    
    def _postprocess_valid_data(self, data) -> list:        
        improve_cf = {}
        for couterfactual in tqdm(data, desc="Improving counterfactuals with binary search"):
            id_key = couterfactual[self.id].values[0]
            original_obs = self.all_data[self.all_data[self.id] == couterfactual[self.id].values[0]].drop(self.config["target"], axis=1).copy()
            cf = couterfactual.copy()
            imrpoved = improve_with_binary_search_by_vectors(original_obs, cf, self.model, self.features_to_use, self.smart_condition)
            improve_cf[id_key] = imrpoved
        return improve_cf
