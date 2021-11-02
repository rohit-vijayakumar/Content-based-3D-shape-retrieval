from matplotlib.pyplot import draw, fill
import pyrender
import trimesh
from trimesh import remesh
import pandas as pd
import numpy as np
from trimesh import Trimesh
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import os
import pickle
import matplotlib.mlab as mlab
import seaborn as sns
import skeletor as sk
from view_data import view_mesh
import copy
from sklearn.preprocessing import MinMaxScaler
from project_statistics import get_outliers
from create_dataset import load_dataset,dir_to_sorted_file_list
from trimesh.repair import fill_holes
from trimesh.points import PointCloud

import random

def standardize(data):
    data_mean = np.mean(data)    
    data_std = np.std(data)
    normalized_data = (data - data_mean)/data_std
    return normalized_data, data_mean, data_std

if __name__=="__main__":
    #before this also rescale mesh with vertices and faces then we can normalize a query object towards same as database objects!
    #features_df = pd.DataFrame(columns = ['id','surface_area', 'compactness','sphericity','volume','diameter','rectangulairty','eccentricity','curvature', 'A3', 'D1', 'D2', 'D3', 'D4'])
    
    # Normalization
    with open('final_features_df.pkl', 'rb') as f:
        data = pickle.load(f)
    normalized_data = data.copy()
    mean_std_list = []

    for column in data.columns[1:8]:
        normalized_feature_data, d_mean, d_std = standardize(data[column])
        mean_std_list.append((d_mean,d_std))
        normalized_data[column] = normalized_feature_data

    normalized_data['diameter'] = 1.0

    normalized_descriptor_data = normalized_data.copy()
    for column in normalized_data.columns[8:]:
        descriptor_data = np.vstack(normalized_data[column])
        column_descriptor_data = descriptor_data/(np.sum(descriptor_data, axis=1)[..., np.newaxis])
        normalized_descriptor_data[column] = column_descriptor_data.tolist()


    normalized_descriptor_data.to_pickle("normalized_final_features_df.pkl")
    np.save('mean_std',np.array(mean_std_list))