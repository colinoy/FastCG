import numpy as np
import pandas as pd
from numpy import linalg as LA
from typing import List

def proc(inst):
    #makes dictionary from dataframe
    inst = inst.to_dict()
    inst_new = {}
    for key, val in inst.items():
        inst_new[key] = [inst[key][k] for k in inst[key].keys()][0]               
    return inst_new

def l1_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-1 distance, sum of absolute difference.
    Parameters
    ----------
    delta: np.ndarray
    Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    # get mask that selects all elements that are NOT zero (with some small tolerance)
    
    absolute_difference = np.abs(delta)
    distance = np.sum(absolute_difference, axis=1, dtype=np.float).tolist()
    return distance 
    


def l0_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-0 norm, number of non-zero entries.
    Parameters
    ----------
    delta: np.ndarray
    Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
        # get mask that selects all elements that are NOT zero (with some small tolerance)
    difference_mask = np.invert(np.isclose(delta, np.zeros_like(delta), atol=1e-05))
    # get the number of changed features for each row
    num_feature_changes = np.sum(
        difference_mask,
        axis=1,
        dtype=np.float,
    )
    distance = num_feature_changes.tolist()
    return distance

def l2_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-2 distance, sum of squared difference.

    Parameters
    ----------
    delta: np.ndarray
    Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    squared_difference = np.square(np.abs(delta))
    distance = np.sum(squared_difference, axis=1, dtype=np.float).tolist()
    return distance 

def linf_distance(delta: np.ndarray) -> List[float]:
    """
    Computes L-infinity norm, the largest change

    Parameters
    ----------
    delta: np.ndarray
    Difference between factual and counterfactual

    Returns
    -------
    List[float]
    """
    absolute_difference = np.abs(delta)
        # get the largest change per row
    largest_difference = np.max(absolute_difference, axis=1)
    distance = largest_difference.tolist()
    return distance

def _get_delta(factual: np.ndarray, counterfactual: np.ndarray) -> np.ndarray:
    """
        Compute difference between original factual and counterfactual

        Parameters
        ----------
        factual: np.ndarray
            Normalized and encoded array with factual data.
            Shape: NxM
        counterfactual: : np.ndarray
            Normalized and encoded array with counterfactual data.
            Shape: NxM

        Returns
        -------
        np.ndarray
    """
    if type(factual) == np.ndarray:
        return counterfactual - factual
    if type(factual) == pd.DataFrame:
        delta = []
        for key in proc(factual).keys():
            delta.append(proc(counterfactual)[key] - proc(factual)[key])
        return np.array(delta).reshape((1,-1))
        
def get_distances(factual: np.ndarray, counterfactual: np.ndarray) -> List[List[float]]:
    """
        Computes distances.
        All features have to be in the same order (without target label).

        Parameters
        ----------
        factual: np.ndarray
            Normalized and encoded array with factual data.
            Shape: NxM
        counterfactual: np.ndarray
            Normalized and encoded array with counterfactual data
            Shape: NxM

        Returns
        -------
        list: distances 1 to 4
    """
    if factual.shape != counterfactual.shape:
        raise ValueError("Shapes of factual and counterfactual have to be the same")
    if len(factual.shape) != 2:
        raise ValueError("Shapes of factual and counterfactual have to be 2-dimensional")

        # get difference between original and counterfactual
    delta = _get_delta(factual, counterfactual)

    d1 = l0_distance(delta)
    d2 = l1_distance(delta)
    d3 = l2_distance(delta)
    d4 = linf_distance(delta)

    return [[d1[i], d2[i], d3[i], d4[i]] for i in range(len(d1))]
    
