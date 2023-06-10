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
from sklearn import tree
from sklearn.tree import export_text



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
        Cutting a subset with fixed features which customer doesn't change.

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
    
    def generate_uniform_points(self,  obs_st, num_points, epsilon):
        """

        Generates synthetic data points for initial estimation.
        Creates with a help of the uniform distribution in the neighborhood of the instance.

        Parameters
        ----------
        num_points : int
            number of points to generate

        Returns
        -------
        pd.DataFrame
            Synthetic dataframe 
        
        """
        dimension = len(self.X_st.columns)
        points = np.random.uniform(obs_st - epsilon, obs_st + epsilon, size=(num_points, dimension))
        X_gen = pd.DataFrame(points, columns = self.X_st.columns)
        y_gen = self.model.predict(X_gen)
        return X_gen, y_gen
    
    
    
    def find_feature_importance_tree(self, X_gen, y_gen):
        """

        Finds the feature importance using decision tree classifier.

        Parameters
        ----------
        data : pd.DataFrame
            The generated dataset for which we want to find the feature importance.

        Returns
        -------
        features to use : list
            List of important features
        features to fix: list
            List of fixed features
        thresholds: list[float]
        
        """
        n_features = self.max_features_to_change
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=n_features)
        clf = clf.fit(X_gen, y_gen)
        ftrs = clf.feature_names_in_
        #important_features = clf.tree_.feature
        features_to_use = [ftrs[k] for k,v in dict(zip(clf.tree_.feature, clf.tree_.threshold)).items() if k >= 0]
        thresholds = [v for k,v in dict(zip(clf.tree_.feature, clf.tree_.threshold)).items() if k >= 0]
        self.features_to_use = features_to_use[0:n_features]
        self.features_to_fix = self.X_st.columns.drop(self.features_to_use)
        self.thresholds = thresholds[0:len(self.features_to_use)]

        return self.features_to_use, self.features_to_fix, self.thresholds



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
        
    def slicing(self, target, n_neighbors):
        """
        Cutting a subset with chosen features and predetermined target class prepared for NN

        Parameters
        ----------
        
        target: int
            target class, usually 0 or 1
        n neighbors: int
            number of neighbors  in NN
            
        
        Returns
        -------
        slice: pd.DataFrame
            Dataframe slice prepared for search
        

        """
        
        d_class = self.all_data[self.model.predict(self.all_data.drop(self.config['target'], axis=1)) == target].index
        data_class_st = self.X_st.loc[self.X_st.index.isin(d_class)]
        obs_ = self.obs_st[self.features_to_fix]
        data_class_ = data_class_st[self.features_to_fix]
        neighbors = NearestNeighbors(n_neighbors= min(n_neighbors, data_class_.shape[0]), algorithm='ball_tree')
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
    
    
    def weitzfeld(self, thresholds, w, n_iter, n_neighbors = 5):
        """
        Choose the counterfactual via weitzfeld algorythm

        Parameters
        ----------
        threshold : list[float]
            the initial point for search
        w : list[float], length = 2
            weights, must be summed up to 1
            w[0] the weight of the instance
            w[1] the weight of target neighbors, works as a proxie for density

        n_iter: int
            number of iterations
        n_neighbors: int
            number of neighbors to be accounterd for
            
        
        Returns
        -------
        dist: float
            weighted sum of distances between the instance, counterfactual and target class
        cand_obs: pd.DataFrame
            counterfactual
        

        """
    
        cand_obs = self.obs_st.copy().reset_index(drop = True)
        cand_obs[self.features_to_use] = thresholds
        if (self.model.predict(cand_obs) == 0):
            print('Please try another initial point')
        cand_dist = euclidean_distances(self.obs_st, cand_obs)[0][0]
        self.slice = self.slicing(1,100)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(self.slice)
        weights = [w[0]] + [w[1]] * n_neighbors
        curr_obs = cand_obs.copy()
        for _ in range(n_iter):
            centers = nbrs.kneighbors(curr_obs, return_distance = True)[1][0]
            points = np.array(self.slice.iloc[centers])
            xj = np.concatenate((np.array(self.obs_st), points))
            #print(xj)
            edi = euclidean_distances(xj, curr_obs)
            #print('edi', edi)
            curr_dist = np.average(edi, weights = weights, axis = 0)[0]
            #print('curr_dist', curr_dist)
            #print('prediction', log_reg.predict(curr_obs))
            if (self.model.predict(curr_obs) == 1) & (curr_dist < cand_dist):
                cand_dist = curr_dist
                cand_obs = curr_obs.copy()
                denom = np.average((1/edi), weights = weights, axis = 0)
                nomin = np.average((xj/edi), weights = weights, axis = 0)
                u = (nomin/denom).reshape((1,len(self.X_st.columns)))
                curr_obs = pd.DataFrame(u, columns = self.X_st.columns)
                for f in self.features_to_fix:
                    curr_obs[f] = self.obs_st[f].values[0]
                #print('curr_obs', curr_obs)
                
            
        
        return cand_obs, cand_dist
    
    # Define the objective function for each agent
    def objective_function(self, curr_obs, xj, weights):
        """

        Finds the objective distance

        Parameters
        ----------
        curr_obs: pd.DataFrame
            the original point
        xj : pd.DataFrame
            points included in the objective distance
        weights: list[float]
        Returns
        -------
        
        curr_dist: float
        
        """
        edi = euclidean_distances(xj, curr_obs)
        curr_dist = np.average(edi, weights = weights, axis = 0)[0]
        return curr_dist
    

    #Define the start point for fine-tuning
    def initial_point(self, features_to_use, thresholds, num_agents = 1):
        """

        Finds the initial estimate

        Parameters
        ----------
        features to use: list
            features to be changed
        thresholds : list[float]
            initial estimate positions, a np.array with number of columns equal to len(features_to_use)
            number of rows equal to number of agents
        num_agents: int
            number of initial estimates
        Returns
        -------
        
        best obs: pd.DataFrame, length =  number of initial agents
        
        """
        best_obs = pd.DataFrame(index = range(num_agents), columns = self.X_st.columns)
        best_obs.loc[[0,]] = self.obs_st.copy().reset_index(drop = True)
        best_obs[features_to_use] = thresholds
        best_obs.fillna(method = 'ffill', inplace = True)
        return best_obs
    
    def beeswarm(self, obs_st, features_to_use, thresholds, w, n_attraction_neighbors = 3, n_repulsion_neighbors = 3, 
    num_agents= 3, num_iterations = 5, inertia_weight = 0.7, cognitive_weight = 0.4, social_weight = 0.3):
        """
        Choose the counterfactual via weitzfeld algorythm

        Parameters
        ----------
        obs_st: pd.DataFrame
            the instance
        features to use: list
            list of features to be changed
        thresholds : list[float], length - number of features to be changed
            the initial point for search
        w : list[float], length = 3
            w[0] the weight of the instance
            w[1] the weight of target neighbors, works as a proxie for density
            w[2] the weight of anti-target neighbors, works as a proxie for density
        n_attraction_neighbors: int
            number of neighbors to be accounted for attraction
        n_repulsion_neighbors: int
            number of neighbors to be accounted for repulsion
        num_agents: int
            number of agents (directions of search)
        num_iterations: int
            number of iterations in pyswarm search
        inertia_weight: float
            inertia weight in pyswarm search
        cognitive_weight: float
            cognitive weight in pyswarm search
        social_weight: float
            social weight in pyswarm search

            
        
        Returns
        -------
        dist: float
            weighted sum of distances between the instance, counterfactual, target class and instance class
        cfs: pd.DataFrame
            counterfactual
        

        """
        slice1 = self.slicing(1,100)
        slice0 = self.slicing(0,100)
        nbrs1 = NearestNeighbors(n_neighbors=n_attraction_neighbors, algorithm='ball_tree').fit(slice1)
        nbrs0 = NearestNeighbors(n_neighbors=n_repulsion_neighbors, algorithm='ball_tree').fit(slice0)
        num_dimensions = len(features_to_use)
        best_obs = self.initial_point(features_to_use, thresholds, num_agents)
        weights = [w[0]] + [w[1]] * n_attraction_neighbors + [w[2]] * n_repulsion_neighbors

        # Initialize the agents' initial positions and fitness values
        epsilon = euclidean_distances(obs_st, best_obs)[0][0]
        positions = thresholds + np.random.uniform(low=-epsilon, high=epsilon, size= (num_agents, num_dimensions))
        velocities = np.zeros((num_agents, num_dimensions))
        best_positions = positions.copy()
        best_dist = np.zeros(num_agents)
        bee = best_obs.copy()
        for i in range(num_agents):
            bee.loc[i,features_to_use] = positions[i]
            best_dist[i] = euclidean_distances(obs_st, bee.loc[[i,]])[0][0]
        
        # Find the global best position and fitness value
        condition = self.model.predict(bee) == 1   #target class
        filtered_arr = best_dist[condition]
        if filtered_arr.any():
            min_index = np.argmin(filtered_arr)
        else:
            print('We cant find any close elements in the target class, try again')
        # Adjust the index to match the original array
        masked_indices = np.where(condition)[0]
        global_best_index = masked_indices[min_index]
        global_best_position = best_positions[global_best_index]
        global_best_dist = best_dist[global_best_index]

        #Iteration  process
        for _ in range(num_iterations):
            print('ITERATION  ', _)
            for i in range(num_agents):
                #print('BEE   ', i)
                centers1 = nbrs1.kneighbors(bee.loc[[i,]], return_distance = True)[1][0]
                points1 = np.array(slice1.iloc[centers1])
                centers0 = nbrs0.kneighbors(bee.loc[[i,]], return_distance = True)[1][0]
                points0 = np.array(slice0.iloc[centers0])
                xj = np.concatenate((np.array(obs_st), points1, points0))
                #print(xj)
                dist = self.objective_function(bee.loc[[i,]], xj, weights)
                if (self.model.predict(bee.loc[[i,]]) == 1) & (dist < best_dist[i]):
                    best_dist[i] = dist
                    best_positions[i] = positions[i]
                    # Update the global best position and fitness value
                    if dist < global_best_dist:
                        global_best_position = positions[i]
                        global_best_dist = dist
                    #print('global dist', global_best_dist)
                    #print('global position', global_best_position)
                    best_obs.loc[i, features_to_use] = positions[i]
                    # Update the agent's velocity
                    #print('best_obs', best_obs)
                    cognitive_component = cognitive_weight * np.random.rand() * (best_positions[i] - positions[i])
                    social_component = social_weight * np.random.rand() * (global_best_position - positions[i])
                    velocities[i] = inertia_weight * velocities[i] + cognitive_component + social_component
                    # Update the agent's position
                    positions[i] += velocities[i]
                    #print('will try positions',positions[i])
                    bee.loc[i, features_to_use] = positions[i]
                elif (self.model.predict(bee.loc[[i,]]) == 0):
                    positions[i] = thresholds[i] + np.random.uniform(low=-epsilon, high=epsilon)
                    #print('will try random positions',positions[i])
                    bee.loc[i, features_to_use] = positions[i]
                    best_dist[i] = euclidean_distances(obs_st, bee.loc[[i,]])[0][0]
            print('best', global_best_position)
            print('__________________________________________')
        
        cfs = obs_st.copy()
        cfs[features_to_use] = global_best_position
        return cfs, global_best_dist


    
    
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
    
    
        
        
    
    
    
    
    