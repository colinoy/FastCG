import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from scipy.stats import norm
from FCG.CounterfactualGenerator import CounterfactualGenerator



class Optika(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class) -> None:
        """
        Initializes the Optika class.

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
        self.max_features_to_change = self.config['max_features_to_change']
        self.normalizer = StandardScaler().fit(all_data.drop(self.config['target'], axis=1))
        #self.features_to_use = self.find_feature_importance_regression()
        
        
    def box(self, obs, n = 1000):
        """
        Cutting a subset with fixed features.

        Parameters
        ----------
        data : pd.DataFrame
            Given data
        config: dict
            configuration
        obs: pd.Series
            given instance
        n: int
            box depth
        
        Returns
        -------
        idx: list
            Necessary indices

        """
        data = self.all_data
        features_box = list(data.columns)
        for c in self.config['features_to_change']:
            features_box.remove(c)
        #o = obs[features_box]            #fixed features part of instance
        cat_cols = set(self.config['categorical columns']).intersection(set(features_box))
        num_cols = set(self.config['numerical columns']).intersection(set(features_box))
        mask = [True] * data.shape[0]
        #if cat_cols:
        #    for col in cat_cols:
        #        mask = mask * (data[col] == o[col].item())
        data_m = data[mask]               #the same category as obs
        idx = data_m.index
        for col in num_cols:
            id_ = pd.Series(data_m[col].sort_values().index)
            i_ = id_.index[id_ == obs.index.item()].item() 
            idx_ = id_[range(max(0,i_-n),min(i_ + n, data.shape[0]))]
            self.idx = list(set(idx).intersection(set(idx_)))     #close to obs
        return self.idx
    
    def pre_process(self, obs):
        """
        Preprocesses the data

        Parameters
        ----------
        None.

        Returns
        -------
        X_st: pd.DataFrame
            standartized dataframe
        y: pd.Series

        """
        
        X = self.all_data.drop(self.config['target'], axis=1)
        self.y = self.all_data[self.config['target']].loc[self.idx,]
        X_st = self.normalizer.transform(X.loc[self.idx,])
        self.X_st = pd.DataFrame(X_st, columns = X.columns, index = self.idx)
        self.obs_st = self.X_st.loc[[obs.index.item(),]]
    

        return self.X_st, self.y, self.obs_st
        




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
    

    def find_feature_importance_regression(self):
        """

        Finds the feature importance using linear regression.

        Parameters
        ----------
        data : pd.DataFrame
            The standartized dataset for which we want to find the feature importance.

        Returns
        -------
        pd.DataFrame
            The feature importance for each feature.
        
        """
        X = self.X_st[self.config["features_to_change"]]
        y = self.y
        n_features = self.max_features_to_change
        linreg = LinearRegression().fit(X, y)
        cs = np.abs(linreg.coef_)
        fs = linreg.feature_names_in_
        dic = list(zip(cs,fs))
        dic.sort(key = lambda x:x[0], reverse = True)
        self.features_to_use = [dic[:n_features][x][1] for x in range(n_features)]
        return self.features_to_use
        
    def slicing(self):
        """
        Cutting a subset with chosen features and right target class

        Parameters
        ----------
        data_box : pd.DataFrame
            Given data after boxing
        data_st : pd.DataFrame
            Standartized data
        model: 
            given model
        obs_st: pd.Series
            standartized instance
        features_fixed: list
            
        
        Returns
        -------
        slice: pd.DataFrame
            Dataframe slice prepared for search
        

        """
        features_fix = list(self.X_st.columns)
        for c in self.features_to_use:
            features_fix.remove(c)
        d_class = self.all_data[self.model.predict(self.all_data.drop(self.config['target'], axis=1)) == 1].index
        data_class_st = self.X_st.loc[self.X_st.index.isin(d_class)]
        obs_ = self.obs_st[features_fix]
        data_class_ = data_class_st[features_fix]
        neighbors = NearestNeighbors(n_neighbors= 500, algorithm='ball_tree')
        neighbors.fit(data_class_)
        indices = neighbors.kneighbors(obs_, return_distance=True) 
        index = [data_class_.iloc[[k,]].index.item() for k in indices[1][0]]
        self.slice = data_class_st.loc[index,]       #fix all features except chosen
        return self.slice

    def optics(self, min_samples, n_neighbors):
        """
        Choose close dense clusters via OPTICS

        Parameters
        ----------
        min_samples : int
            number of points in 'dense' cluster
        slice : pd.DataFrame
            data slice
        obs_st: pd.Series
            standartized instance
        n_neighbors: int
            number of target clusters
            
        
        Returns
        -------
        radius: float
            distances
        data_res: pd.DataFrame
            Dataframe of neighbors
        

        """
        
        opt = OPTICS(min_samples = min_samples).fit(self.slice)
        labels = opt.labels_
        self.n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        self.n_noise_ = list(labels).count(-1)
        self.slice['labels'] = labels
        data_centr = self.slice.groupby(['labels']).mean()
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(data_centr)
        self.radius = nbrs.kneighbors(self.obs_st, return_distance = True)[0][0]
        centers = nbrs.kneighbors(self.obs_st, return_distance = True)[1][0]
        self.data_res = data_centr.loc[centers,].reset_index()
        #data_res = data_centr.loc[slice['labels'].isin(cf_labels)]
        return self.radius, self.data_res
    
    
    def generate(self):
        cfs = pd.DataFrame(columns = self.X_st.columns)
        dist = []
        n_neighbors = []
        neigh = NearestNeighbors(radius = 1, algorithm='ball_tree')
        neigh.fit(self.X_st)
        for i in list(self.data_res.index):
            res_st = self.obs_st.copy()
            b = res_st.loc[:, self.features_to_use]
            c = self.data_res.loc[i,self.features_to_use]
            for x in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5]:
                res_st[self.features_to_use] = b + x*(c-b)
                res = pd.DataFrame(self.normalizer.inverse_transform(res_st), columns = self.X_st.columns)
                if self.model.predict(res) == 1:
                    d = euclidean_distances(self.obs_st, res_st)
                    dist.append(d[0][0])
                    rng = neigh.radius_neighbors(res_st, radius = 1.5)
                    n_neighbors.append(len(rng[0][0]))
                    cfs = pd.concat([cfs, res], axis=0)

        
        return cfs, dist, n_neighbors

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
    