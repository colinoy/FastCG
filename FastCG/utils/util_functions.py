import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize
from FastCG.utils.logger import Logger

def find_centroid(data, n_clusters=100, random_state=0, features_pca=None):
    # n_clusters is min (0.1*len(data), n_clusters), sanity in case of small data size
    n_clusters = min(int(0.1*len(data)), n_clusters)
    kmean = KMeans(n_clusters=n_clusters,
                        random_state=random_state).fit(data)
    return pd.DataFrame(kmean.cluster_centers_, columns=features_pca), kmean.labels_


def find_feature_improtance_Anova(data, n_features, features_to_change, target):

    X = data[features_to_change]
    #if X has categorical features, we need to convert them to numerical
    if X.select_dtypes(include=['object', 'category']).shape[1] > 0:
        #label encoding
        X = X.apply(lambda x: pd.factorize(x)[0])
    y = data[target]
    anova = SelectKBest(score_func=f_classif, k=n_features)
    fit = anova.fit(X, y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    # naming the dataframe columns
    featureScores.columns = ['Features', 'Score']
    topn = featureScores.nlargest(n_features, 'Score')
    return topn["Features"].values.tolist()


def choose_pca_components(data, variance=0.85, normalizer=None):
    pca = PCA(n_components=variance)
    if normalizer is None:
        Logger.info("Assuming data is already normalized")
        pca.fit(data)
    else:
        Logger.info("Normalizing data")
        pca.fit(normalizer.transform(data))
    return pca.n_components_

def convert_to_pca(data, pca, normalizer, features_to_use, features_pca):
        original_data = data.copy()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        features_to_use = [feature for feature in features_to_use if feature not in categorical_features]
        if normalizer is not None:
            data = normalizer.transform(data[features_to_use])
        else:
            data = data[features_to_use]
        data = pca.transform(data)
        data = pd.DataFrame(data = data, columns = features_pca)
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(features_to_use, axis=1)], axis=1).drop_duplicates()
        return data 

def convert_from_pca(data, original_data, pca,normalizer, features_to_use, features_pca):
        # original_data = self.all_data[self.all_data[self.config["ID"]] == data[self.config["ID"]].values[0]].copy()
        categorical_features = data.select_dtypes(include=['object', 'category']).columns
        features_to_use = [feature for feature in features_to_use if feature not in categorical_features]
        data = pca.inverse_transform(data[features_pca]).reshape(1, -1)
        data = pd.DataFrame(data = data, columns = features_to_use) 
        data = normalizer.inverse_transform(data)
        data = pd.DataFrame(data = data, columns = features_to_use)
        data = data.reset_index(drop=True)
        original_data = original_data.reset_index(drop=True)
        data = pd.concat([data, original_data.drop(features_to_use, axis=1)], axis=1).dropna()
        data = data[original_data.columns]
        return data

def find_feature_direction(data_matching_condition, features, target):
        features_increase_only = []
        features_decrease_only = []
        for feature in features:
            if data_matching_condition[feature].corr(data_matching_condition[target], method='pearson') > 0:
                features_increase_only.append(feature)
            else:
                features_decrease_only.append(feature)
        return features_increase_only, features_decrease_only


def obs_distance(counterfactual, obs, target_colunm):
        if target_colunm in counterfactual.columns:
            counterfactual = counterfactual.drop(target_colunm, axis=1)
        if target_colunm in obs.columns:
            obs = obs.drop(target_colunm, axis=1)
        categorical_features = counterfactual.select_dtypes(include=['object', 'category']).columns
        features_to_use = [feature for feature in counterfactual.columns if feature not in categorical_features]
        counterfactual_num = counterfactual[features_to_use]
        obs_num = obs[features_to_use]
        distance_between_categorical_features = [0 if counterfactual[feature].values[0] == obs[feature].values[0] else 1 for feature in categorical_features]
        return euclidean_distances(counterfactual_num, obs_num)[0][0] + sum(distance_between_categorical_features)


def calculate_point(start,direction,step):
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
        categorical_features = start.select_dtypes(include=['object', 'category']).columns
        features_to_use = [feature for feature in start.columns if feature not in categorical_features]

        return start[features_to_use] + step * direction

def calculate_distance(point1,point2):
    categorical_features = point1.select_dtypes(include=['object', 'category']).columns
    features_to_use = [feature for feature in point1.columns if feature not in categorical_features]
    return np.linalg.norm(point1[features_to_use].values - point2[features_to_use].values)

def calculate_direction(point1,point2):
    categorical_features = point1.select_dtypes(include=['object', 'category']).columns
    features_to_use = [feature for feature in point1.columns if feature not in categorical_features]
    sub_point = point2[features_to_use].values - point1[features_to_use].values
    norm = sub_point/calculate_distance(point1,point2)
    # norm_normlized = normalize(norm)
    return norm

def improve_with_binary_search_by_vectors(original_obs, genrated_cf, model, features_to_use, smart_condition):
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
    cols = original_obs.select_dtypes(exclude=['object', 'category']).columns
    categorical_features = original_obs.select_dtypes(include=['object', 'category']).columns
    features_to_use = [feature for feature in features_to_use if feature in cols]
    categorical_data = original_obs[categorical_features]
    distance = calculate_distance(original_obs,genrated_cf) 
    direction = calculate_direction(original_obs,genrated_cf) 
    
    start_distance = -distance
    end_distance = distance
    try:
        while (end_distance-start_distance).sum() > distance*0.001: 
            mid_point = calculate_point(original_obs,direction,(start_distance+end_distance)/2)
            mid_point = mid_point[cols]
            mid_point[features_to_use] = mid_point[features_to_use].astype(int)
            #combine the categorical and numerical features 
            mid_point = pd.concat([mid_point, original_obs[categorical_features]], axis=1)
            if smart_condition.check(model.predict(mid_point.drop("ID", axis=1))[0]):
                end_distance = (start_distance+end_distance)/2
            else:
                start_distance = (start_distance+end_distance)/2
    except Exception as e:
        Logger.exception("Error in binary search: {}".format(e))
        Logger.exception(mid_point)
        return None
    end_distance += distance*0.01
    final_point = calculate_point(original_obs,direction,end_distance)
    #merge the final point with the categorical features 
    final_point = pd.concat([final_point, categorical_data], axis=1)
    return final_point