
from matplotlib.pyplot import draw, fill
import pyrender
import glob
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
import re
import random
import pickle
from scipy.stats import wasserstein_distance
from utils import*

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

import warnings
warnings.filterwarnings("ignore")


with open('query_database.pkl', 'rb') as f:
    query_database = pickle.load(f)


DB_DIRECTORY = r"full_normalized_benchmark\**\*.off"
mesh_files = list(glob.glob(DB_DIRECTORY,recursive=True))
mesh_files.sort(key=natural_keys)
mean_std = np.load('mean_std.npy')


# In[5]:


K = 6
for i,mesh_file in enumerate(mesh_files):
    features = []
    features_df = pd.DataFrame(columns = ['id','surface_area','volume','compactness','sphericity','diameter','rectangulairty','eccentricity','curvature', 'A3', 'D1', 'D2', 'D3', 'D4'])
    if (i==221 or i==700 or i==1313):
        f = 0
        mesh = trimesh.load(mesh_file,force='mesh')
        #view_mesh(mesh)
        
        # Curvature 
        data = discrete_gaussian_curvature_measure(mesh,mesh.vertices, 0.1)
        local_weight = 0
        global_weight = 1
        scaler = MinMaxScaler()
        norm_data = np.array(local_weight*data - global_weight*data).reshape(-1,1)
        norm_data = scaler.fit_transform(norm_data)
        norm_hist, _ = np.histogram(norm_data,bins=8)
        curvature = norm_hist
        norm = matplotlib.colors.Normalize(vmin=(local_weight*min(data))-(global_weight*40), vmax=(local_weight*max(data))+ (global_weight*30), clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.turbo)
        node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(data)]
        mesh.visual.vertex_colors = node_color
        #mesh.show()

        # Scalar Features
        pc_mesh = PointCloud(mesh.vertices).convex_hull
        diameter = np.max(pc_mesh.bounds[1]-pc_mesh.bounds[0])
        features.append(i)

        normalized_area, f = standardize(mesh.area,f, mean_std)
        features.append(normalized_area)

        volume =  pc_mesh.volume
        thresh = 0.00823
        if (volume < thresh):
            volume = thresh

        normalized_volume, f = standardize(volume, f, mean_std)
        features.append(normalized_volume)

        compactness = (pc_mesh.area**3)/ (36* np.pi*(volume**2))
        sphericity = 1/compactness

        normalized_compactness, f = standardize(compactness, f, mean_std)
        features.append(normalized_compactness)
        normalized_sphericity, f = standardize(sphericity, f, mean_std)
        features.append(normalized_sphericity)

        features.append(diameter)
        f+=1

        rectangularity = volume / mesh.bounding_box.volume
        normalized_rectangularity, f = standardize(rectangularity, f, mean_std)
        features.append(normalized_rectangularity)

        eccentricity  = abs(mesh.principal_inertia_components[0] / mesh.principal_inertia_components[2])
        normalized_eccentricity, f = standardize(eccentricity, f, mean_std)
        features.append(normalized_eccentricity)

        normalzied_curvature = curvature/np.sum(curvature)
        features.append(normalzied_curvature)

        # Shape descriptors
        A3 = []
        D1 = []
        D2 = []
        D3 = []
        D4 = []
        v_bary = mesh.centroid
        c = 0
        for j in range (50000):
            n1 = random.randint(0,len(mesh.vertices)-1)
            n2 = random.randint(0,len(mesh.vertices)-1)
            n3 = random.randint(0,len(mesh.vertices)-1)
            n4 = random.randint(0,len(mesh.vertices)-1)

            a = []
            a.append(n1)
            a.append(n2)
            a.append(n3)
            a.append(n4)
            if(len(np.unique(a))!=len(a)):
                j-=1
                continue
            v1 = mesh.vertices[n1] 
            v2 = mesh.vertices[n2] 
            v3 = mesh.vertices[n3]
            v4 = mesh.vertices[n4]

            # A3 Angle
            vec1 = v2-v1
            vec2 = v3-v1
            vec3 = v4-v1

            norm_v1 = np.linalg.norm(vec1)
            norm_v2 = np.linalg.norm(vec2)
            if ((norm_v1* norm_v2) ==0):
                j-=1
                continue
            if math.isnan(np.dot(vec1,vec2) / (norm_v1* norm_v2) ):
                j-=1
                continue
            angle = (np.rad2deg(np.arccos( np.dot(vec1,vec2) / (norm_v1* norm_v2) )))
            if math.isnan(angle):
                j-=1
                continue
            A3.append(angle)

            distance = np.linalg.norm(v1-v_bary)
            D1.append(distance)

            distance_2 = np.linalg.norm(v1-v2)
            D2.append(distance_2)

            crosses = np.array([np.cross(vec1,vec2)])
            area = (np.sum(crosses**2, axis=1)**.5) * .5
            distance_3 = min(np.sum(area), 0.1)
            D3.append(distance_3)

            prod = np.linalg.norm( np.dot(np.cross(vec1,vec2),vec3))
            volume = (1/6)*prod
            cube_root = volume**(1/3)
            D4.append(cube_root)

        A3_descriptor, x = np.histogram(A3,bins=8)
        normalzied_A3_descriptor = A3_descriptor/np.sum(A3_descriptor)
        features.append(normalzied_A3_descriptor)

        D1_descriptor, x = np.histogram(D1,bins=8)
        normalzied_D1_descriptor = D1_descriptor/np.sum(D1_descriptor)
        features.append(normalzied_D1_descriptor)            

        D2_descriptor, x = np.histogram(D2,bins=8)
        normalzied_D2_descriptor = D2_descriptor/np.sum(D2_descriptor)
        features.append(normalzied_D2_descriptor)              

        D3_descriptor, x = np.histogram(D3,bins=8)
        normalzied_D3_descriptor = D3_descriptor/np.sum(D3_descriptor)
        features.append(normalzied_D3_descriptor)             

        D4_descriptor, x = np.histogram(D4,bins=8)
        normalzied_D4_descriptor = D4_descriptor/np.sum(D4_descriptor)
        features.append(normalzied_D4_descriptor)
        
        features_df.loc[len(features_df)] = features
    
        query = expand_df(features_df)
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(np.array(query_database.iloc[:,1:]), np.array(query_database['class']))

        knn_query = np.array(query)
        pred_list = knn.kneighbors(knn_query, K)[1][0]
        print(i, pred_list) 
        get_query(pred_list)




