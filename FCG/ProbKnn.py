from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from functools import partial
from scipy.spatial.distance import minkowski
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

class ProbKnn(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class) -> None:
        """
        Initializes the ProbKnn class.

        Parameters
        ----------
        all_data : pd.DataFrame
            The entire dataset. 
        model : sklearn model
            The model to be explained.
        config : dict
            The configuration dictionary, containing the following keys: 
                - features_to_change: list of features that can be changed
                - target: the target feature
        target_class : int 
            The target class for which we want to generate counterfactuals.

        Returns
        -------
        None.

        """
        super().__init__(all_data, model, config, target_class)  
        self.nbrs = None
        self.features_to_use = self.find_feature_improtance_Anova(self.all_data)
        self.normalizer = StandardScaler().fit(self.all_data[self.features_to_use])
        self.n_components = self.choose_pca_components()
        self.pca = PCA(n_components=self.n_components).fit(self.normalizer.transform(self.all_data[self.features_to_use]))
        self.features_pca = ['principal component ' + str(i) for i in range(1, self.n_components+1)]


    def pre_process(self):
        """
        Preprocesses the data to be used in the KNN algorithm.

        Parameters
        ----------
        None.

        Returns
        -------
        None.

        """
        weights = [] 
        self.data_class = self.all_data[self.model.predict(self.all_data.drop(self.config["target"], axis=1)) == self.target_class]
        self.data_with_weights = self.convert_to_pca(self.data_class.copy())
        self.data_centroid = self.find_centroid(self.data_with_weights.drop(self.config["target"], axis=1).copy())
        self.nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(self.data_centroid)


    
    def find_centroid(self, data):
        """

        Finds the centroid of a given dataset using Density-based clustering

        Parameters
        ----------
        data : pd.DataFrame
            The dataset for which we want to find the centroid.

        Returns
        -------
        pd.DataFrame
            The centroid of the given dataset.

        """
        self.kmean = KMeans(n_clusters=100, random_state=0).fit(data.drop(self.features_pca, axis=1))
        return pd.DataFrame(self.kmean.cluster_centers_, columns=data.drop(self.features_pca, axis=1).columns)

        

    def compute_weights_from_probabilites(self, data):
        """
        Computes the weights for each feature based on the probabilities of the model.

        Parameters
        ----------
        data : pd.DataFrame
            The data to compute the weights for.

        Returns
        -------
        data : pd.DataFrame
            The data with the weights.
        

        """
        scaler = StandardScaler()
        for feature in data.drop(self.config["target"], axis=1).columns:
            if feature == self.config["ID"]:
                continue
            if feature not in self.features_pca:
                data[feature] = scaler.fit_transform(data[feature].values.reshape(-1,1))
        return data

    def distance(self, counterfactual, obs):
        """

        Computes the distance between the counterfactual and the original observation.

        Parameters
        ----------
        counterfactual : pd.DataFrame
            The counterfactual found using the KNN algorithm.
        obs : pd.DataFrame
            The original observation.

        Returns
        -------
        float
            The distance between the counterfactual and the original observation.
        
        """
        if self.config["target"] in counterfactual.columns:
            counterfactual = counterfactual.drop(self.config["target"], axis=1)
        if self.config["target"] in obs.columns:
            obs = obs.drop(self.config["target"], axis=1)
        return euclidean_distances(counterfactual, obs)[0][0]
    

    def knn_on_cluster(self, obs):
        obs_original = obs[self.data_with_weights.drop(self.config["target"], axis=1).columns]
        obs = obs[self.data_with_weights.drop(self.config["target"], axis=1).drop(self.features_pca, axis=1).columns]
        indices = self.nbrs.kneighbors(obs, return_distance=False) 
        index = []
        for indice in indices[0]:
            cluster_members = [j for j, x in enumerate(self.kmean.labels_) if x == indice]
            max_index = min(len(cluster_members), 10)
            for member in cluster_members[:max_index]:
                idx = self.data_with_weights.iloc[member].name
                if idx not in index:
                    index.append(idx)
        
        weights = []
        data_cluster = self.data_with_weights.loc[index] 
        for feature in data_cluster.drop(self.config["target"], axis=1).columns: 
            if feature == self.config["ID"]:
                weights.append(0)  
            elif feature not in self.features_pca: 
                weights.append(1)
            else:
                weights.append(0)
        w_minkowski = partial(minkowski, p=2, w=weights)
        nbrs = NearestNeighbors(n_neighbors=2, metric=w_minkowski).fit(data_cluster.drop(self.config["target"], axis=1))
        indices = nbrs.kneighbors(obs_original, return_distance=False)
        return indices[0]
        

    def generate_single_counterfactual(self, obs):
        """
        Generates a single counterfactual for a given observation.

        Parameters
        ----------
        obs : pd.DataFrame
            The observation for which we want to generate a counterfactual.

        Returns
        -------
        pd.DataFrame
            The counterfactual for the given observation.

        """
        indices = self.knn_on_cluster(obs)
        obs_original = self.convert_to_original(obs)
        cf = {}
        for idx in indices:
            counterfactual = obs.copy()
            for feature in self.features_pca: 
                counterfactual[feature] = self.data_with_weights[feature].iloc[idx]
            counterfactual = self.convert_to_original(counterfactual)
            # for feature in self.config["increase_features"]:
            #     if counterfactual[feature].values[0] < obs_original[feature].values[0]:
            #         counterfactual[feature] = obs_original[feature].values[0]*1.5
            if self.model.predict(counterfactual.drop(self.config["target"], axis=1)) == self.target_class:
                # return counterfactual
                distance = self.distance(counterfactual.copy(), obs.copy())
                if distance not in cf:
                    cf[distance] = counterfactual.copy()
        if cf:
            return cf[min(cf.keys())]
        return None 
    

    def find_feature_improtance_Anova(self, data, n_features=2):
        """

        Finds the feature importance using Anova.

        Parameters
        ----------
        data : pd.DataFrame
            The dataset for which we want to find the feature importance.

        Returns
        -------
        pd.DataFrame
            The feature importance for each feature.
        
        """
        X = data[self.config["features_to_change"]]
        y = data[self.config["target"]]
        anova = SelectKBest(score_func=f_classif, k=n_features)
        fit = anova.fit(X, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(X.columns)
        featureScores = pd.concat([dfcolumns,dfscores],axis=1)
        featureScores.columns = ['Features','Score']  #naming the dataframe columns
        topn = featureScores.nlargest(n_features,'Score')
        return topn["Features"].values.tolist()
        


    def generate_mulitple_counterfactuals(self, data_to_genrate):
        """

        Generates multiple counterfactuals for a given dataset.
        
        Parameters
        ----------
        data_to_genrate : pd.DataFrame
            The dataset for which we want to genersate counterfactuals.

        Returns
        -------
        counterfactual_list : dict
            A dictionary containing the counterfactuals for each observation.
        
        """
        counterfactual_list = {}
        data_to_genrate = data_to_genrate[self.model.predict(data_to_genrate) != self.target_class]
        data_to_genrate = self.convert_to_pca(data_to_genrate)
        data_to_genrate = data_to_genrate.fillna(0)
        for _, obs in data_to_genrate.iterrows():
            obs = obs.to_frame().T
            counterfactual = self.generate_single_counterfactual(obs) 
            if counterfactual is not None:
                obs = self.convert_to_original(obs)
                counterfactual_using_binary_search = self.improve_with_binary_search_by_vectors(obs, counterfactual)
                counterfactual_using_binary_search[self.features_to_use] = counterfactual_using_binary_search[self.features_to_use].astype(int)
                counterfactual_list[obs[self.config["ID"]].values[0]] = counterfactual_using_binary_search
        return counterfactual_list
    
    def choose_pca_components(self):
        """
        Finds the number of PCA components that explain 80% of the variance.

        Returns
        -------
        int
            The number of PCA components that explain 80% of the variance.

        """
        pca = PCA(n_components=0.95)
        pca.fit(self.normalizer.transform(self.all_data[self.features_to_use]))
        return pca.n_components_ 
        
        
    def convert_to_pca(self, data):
        """
        Converts the data to the PCA space.

        Parameters
        ----------
        data : pd.DataFrame
            The data to convert to the PCA space.

        Returns
        -------
        data : pd.DataFrame
            The data in the PCA space.
        """
        original_data = data.copy()
        data = self.normalizer.transform(data[self.features_to_use])
        data = self.pca.transform(data)
        data = pd.DataFrame(data = data, columns = self.features_pca)
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(self.features_to_use, axis=1)], axis=1).dropna()
        return data 
    
    def convert_to_original(self, data):
        """
        Converts the data from the PCA space to the original space.

        Parameters
        ----------
        data : pd.DataFrame
            The data to convert to the original space.

        Returns
        -------
        data : pd.DataFrame
            The data in the original space.        
        """
        original_data = self.all_data[self.all_data[self.config["ID"]] == data[self.config["ID"]].values[0]].copy()
        data = self.pca.inverse_transform(data[self.features_pca]).reshape(1, -1)
        data = pd.DataFrame(data = data, columns = self.features_to_use) 
        data = self.normalizer.inverse_transform(data)
        data = pd.DataFrame(data = data, columns = self.features_to_use)
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(self.features_to_use, axis=1)], axis=1).dropna()
        data = data[self.all_data.columns]
        return data
    
    def calculate_point(self,start,direction,step):
        """
        Calculates a point in a given direction and distance from a given point.

        Parameters
        ----------  
        start : np.array
            The starting point.
        direction : np.array
            The direction in which we want to move.

        Returns
        -------
        np.array
            The point in the given direction and distance from the starting point.

        """
        return start + direction*step

    def calculate_distance(self,point1,point2):
        return np.linalg.norm(point1-point2)

    def calculate_direction(self,point1,point2):
        return (point2-point1)/np.linalg.norm(point2-point1)

    def improve_with_binary_search_by_vectors(self,start_point,end_point):
        """
        Improves the counterfactual found using the knn method by using binary search. 

        Parameters
        ----------
        start_point : pd.DataFrame
            The starting point - the original observation. 
        end_point : pd.DataFrame
            The end point - the counterfactual found using the knn method.

        Returns
        -------
        pd.DataFrame
            The improved counterfactual.

        """
        
        distance = self.calculate_distance(start_point,end_point) 
        direction = self.calculate_direction(start_point,end_point) 
        
        start_distance = -distance
        end_distance = distance

        while (end_distance-start_distance).sum() > distance*0.001: 
            mid_point = self.calculate_point(start_point,direction,(start_distance+end_distance)/2)
            mid_point = mid_point[self.all_data.drop(self.config["target"], axis=1).columns]
            mid_point[self.features_to_use] = mid_point[self.features_to_use].astype(int)
            if self.model.predict(mid_point) == self.target_class:
                end_distance = (start_distance+end_distance)/2
            else:
                start_distance = (start_distance+end_distance)/2

        return self.calculate_point(start_point,direction,(start_distance+end_distance)/2)