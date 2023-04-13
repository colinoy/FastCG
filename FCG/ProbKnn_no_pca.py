from FCG.CounterfactualGenerator import CounterfactualGenerator
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from functools import partial
from scipy.spatial.distance import minkowski
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
import numpy as np
import pandas as pd
from scipy.stats import norm

class ProbKnn_no_pca(CounterfactualGenerator):
    def __init__(self, all_data, model, config, target_class):
        super().__init__(all_data, model, config, target_class) #initialize the parameters for the counterfactual generator from the parent class
        #initialize the parameters for the counterfactual generator including normalizer and pca 
        self.nbrs = None
        self.normalizer = StandardScaler().fit(self.all_data[self.config["features_to_change"]])  
        self.scaler = MinMaxScaler()


    def pre_process(self):
        weights = [] 
        self.data_class = self.all_data[self.model.predict(self.all_data.drop(self.config["target"], axis=1)) == self.target_class]
        # self.data_pca = self.convert_to_pca(self.data_class)
        #normalize the feature that are not in the features_to_change list
        self.data_with_weights = self.compute_weights_from_probabilites(self.data_class, to_print=False)
        for feature in self.data_with_weights.drop(self.config["target"], axis=1).columns: #for loop to generate the weights for the weighted minkowski distance
            if feature == "Loan ID" or feature == "Customer ID":
                weights.append(0) 
            elif feature not in self.config["features_to_change"]: 
             #if the feature is not in features_to_change, give it a weight of 1 (we want to keep the value of these features the same)
                weights.append(1)
            elif feature in self.config["features_to_change"]: 
                weights.append(0)
            else:
                weights.append(0)
        w_minkowski = partial(minkowski, p=2, w=weights) #weighted minkowski distance. 
        #finding the k closest neighbors of the instance using the weighted minkowski distance (w is n by m matrix, where n is the number of instances and m is the number of features)
        self.nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree', metric=w_minkowski).fit(self.data_with_weights.drop(self.config["target"], axis=1))


    def compute_weights_from_probabilites(self, data, to_print=False):
        """
        this function recive obserevation, and for each feature compute their probability in the desired class. 
        the function return a list of the probabilities of each feature. 
        The goal is to give higher weight to features values with lower probability in the desired class.            
        """
        #get the probability of the feature value in the desired class, if the probability is very low, give the feature a high weight (bacause its rare in the desired class)
        #if the probability is high, give the feature a low weight (because its common in the desired class) 
        #normalize the entire data to be between 0 and 1 
        for feature in data.columns:
            if feature == "Loan ID" or feature == "Customer ID" or feature == self.config["target"]:
                continue
            data[feature] = (data[feature] - data[feature].min()) / (data[feature].max() - data[feature].min())
            # if feature in self.config["features_to_change"]:
            #     data[feature + "_weight"] = 1 + data[feature]
        return data.fillna(0)

        
    
    def generate_single_counterfactual(self, obs):
        """ This function is used to generate a single counterfactual for a single instance. 
            The function takes the instance and find the nearest neighbors of the instance, and assign for the instance the feature values of the nearest neighbor.
            After that, the function check if the counterfactual is in the desired class. If it is, the function return the counterfactual. 
            If it is not, the function will go over all the nearest neighbors and check if the counterfactual is in the desired class. """
        obs = obs[self.data_with_weights.drop(self.config["target"], axis=1).columns]
        indices = self.nbrs.kneighbors(obs, return_distance=False) #get the indices of the nearest neighbors
        for indice in indices[0]: #for loop over the indices of the nearest neighbors
            counterfactual = obs.copy()
            for feature in self.config["features_to_change"]: #for loop over the features to change
                counterfactual[feature] = self.data_with_weights.iloc[indice][feature] #assign the feature value of the nearest neighbor to the instance
                #data with weights is normalized, so we need to denormalize the counterfactual 
                counterfactual[feature] = counterfactual[feature] * (self.all_data[feature].max() - self.all_data[feature].min()) + self.all_data[feature].min()
            # counterfactual = counterfactual.to_frame().T
            # counterfactual = counterfactual.drop(counterfactual.filter(regex='_weight').columns, axis=1)
            
            if self.model.predict(counterfactual) == self.target_class: #check if the counterfactual is in the desired class
                return counterfactual
        return None 


    def generate_mulitple_counterfactuals(self, data_to_genrate):
        """
        This function is used to generate multiple counterfactuals for multiple instances.
        This function get as input a dataframe with multiple instances and generate a counterfactual for each instance that is not in the desired class.
        """
        counterfactual_list = {}
        data_to_genrate = data_to_genrate[self.model.predict(data_to_genrate.drop(self.config["target"], axis=1)) != self.target_class]
        # data_to_genrate_norm = self.compute_weights_from_probabilites(data_to_genrate)
        for _, obs in data_to_genrate.iterrows():
            obs = obs.to_frame().T.drop(self.config["target"], axis=1)
            # display(obs) 
            counterfactual = self.generate_single_counterfactual(obs) #generate a counterfactual for a single instance
            if counterfactual is not None: #if the counterfactual is not None, add the counterfactual to the counterfactual list and try to improve the counterfactual using binary search
                #print the counterfactual weight
                # display(data_to_genrate[data_to_genrate["Loan ID"] == obs["Loan ID"].values[0]].filter(regex='_weight'))
                counterfactual = self.improve_with_binary_search_by_vectors(obs, counterfactual)
                counterfactual_list[obs["Loan ID"].values[0]] = counterfactual
        return counterfactual_list
    
    

    def choose_pca_components(self): #choose the number of pca components to use 
        pca = PCA(n_components=0.9)
        pca.fit(self.normalizer.transform(self.all_data[self.config["features_to_change"]]))
        return pca.n_components_ 
        
        
    def convert_to_pca(self, data):
        original_data = data.copy()
        data = self.normalizer.transform(data[self.config["features_to_change"]])
        data = self.pca.transform(data)
        data = pd.DataFrame(data = data, columns = self.features_pca)
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(self.config["features_to_change"], axis=1)], axis=1).dropna()
        return data 
    
    def convert_to_original(self, data):
        original_data = self.all_data[self.all_data["Loan ID"] == data["Loan ID"].values[0]].copy()
        data = self.pca.inverse_transform(data[self.features_pca]).reshape(1, -1)
        data = pd.DataFrame(data = data, columns = self.config["features_to_change"]) 
        data = self.normalizer.inverse_transform(data)
        data = pd.DataFrame(data = data, columns = self.config["features_to_change"])
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(self.config["features_to_change"], axis=1)], axis=1).dropna()
        data = data[self.all_data.columns]
        # print(data)
        return data
    
    def calculate_point(self,start,direction,step):
        return start + direction*step

    def calculate_distance(self,point1,point2):
        return np.linalg.norm(point1-point2)

    def calculate_direction(self,point1,point2):
        return (point2-point1)/np.linalg.norm(point2-point1)

    def improve_with_binary_search_by_vectors(self,start_point,end_point):
        """
        This function is used to improve the counterfactual using binary search. 
        The input of this function is the original instance and the counterfactual and it aim to improve the counterfactual by finding the point that is in the desired class and is the closest to the original instance.
        """
        distance = self.calculate_distance(start_point,end_point) #calculate the distance between the original instance and the counterfactual
        direction = self.calculate_direction(start_point,end_point) #calculate the direction between the original instance and the counterfactual 

        start_distance = 0
        end_distance = distance

        while (end_distance-start_distance).sum() > 0.0001: #while the distance between the start point and the end point is bigger than 0.0001, continue to improve the counterfactual
            mid_point = self.calculate_point(start_point,direction,(start_distance+end_distance)/2)
            mid_point = mid_point[self.all_data.drop(self.config["target"], axis=1).columns]
            mid_point[self.config["features_to_change"]] = mid_point[self.config["features_to_change"]].astype(int)
            if self.model.predict(mid_point) == self.target_class: #if the mid point is in the desired class, move the end point to the mid point
                end_distance = (start_distance+end_distance)/2
            else:
                start_distance = (start_distance+end_distance)/2 #if the mid point is not in the desired class, move the start point to the mid point

        return self.calculate_point(start_point,direction,(start_distance+end_distance)/2)
    

    def Calculate_the_relative_changes_of_the_neighbors(self, obs, counterfactual):
        """
        This function is used to calculate the relative changes of the neighbors of the original instance and the counterfactual.
        """
        relative_changes = {}
        for feature in self.config["features_to_change"]:
            if obs[feature].values[0] == 0:
                relative_changes[feature] = 0
            relative_changes[feature] = (counterfactual[feature].values[0] - obs[feature].values[0])/obs[feature].values[0]
        # Calculate the weight vector as the mean of the relative changes
        weight_vector = np.array(list(relative_changes.values())) 
        weight_vector = weight_vector/np.linalg.norm(weight_vector)
        print(weight_vector)
        return weight_vector
    

    def distance(self, obs, counterfactual, weight_vector):
        distance = 0
        mAD = np.median(np.abs(obs - np.median(obs, axis=0)), axis=0) 
        for j in range(len(obs)):
            distance += np.abs(obs[j] - counterfactual[j]) / (mAD[j] * weight_vector[j])
        return distance