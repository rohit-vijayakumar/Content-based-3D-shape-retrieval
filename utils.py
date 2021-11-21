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
from scipy.spatial import distance
from scipy.stats import wasserstein_distance

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def is_sequence(obj):
    """
    Check if an object is a sequence or not.
    Parameters
    -------------
    obj : object
      Any object type to be checked
    Returns
    -------------
    is_sequence : bool
        True if object is sequence
    """
    seq = (not hasattr(obj, "strip") and
           hasattr(obj, "__getitem__") or
           hasattr(obj, "__iter__"))

    # check to make sure it is not a set, string, or dictionary
    seq = seq and all(not isinstance(obj, i) for i in (dict,
                                                       set,
                                                       basestring))

    # PointCloud objects can look like an array but are not
    seq = seq and type(obj).__name__ not in ['PointCloud']

    # numpy sometimes returns objects that are single float64 values
    # but sure look like sequences, so we check the shape
    if hasattr(obj, 'shape'):
        seq = seq and obj.shape != ()

    return seq

def is_shape(obj, shape, allow_zeros=False):
    """
    Compare the shape of a numpy.ndarray to a target shape,
    with any value less than zero being considered a wildcard
    Note that if a list-like object is passed that is not a numpy
    array, this function will not convert it and will return False.
    Parameters
    ------------
    obj :   np.ndarray
      Array to check the shape on
    shape : list or tuple
      Any negative term will be considered a wildcard
      Any tuple term will be evaluated as an OR
    allow_zeros: bool
      if False, zeros do not match negatives in shape
    Returns
    ---------
    shape_ok : bool
      True if shape of obj matches query shape
    Examples
    ------------------------
    In [1]: a = np.random.random((100, 3))
    In [2]: a.shape
    Out[2]: (100, 3)
    In [3]: trimesh.util.is_shape(a, (-1, 3))
    Out[3]: True
    In [4]: trimesh.util.is_shape(a, (-1, 3, 5))
    Out[4]: False
    In [5]: trimesh.util.is_shape(a, (100, -1))
    Out[5]: True
    In [6]: trimesh.util.is_shape(a, (-1, (3, 4)))
    Out[6]: True
    In [7]: trimesh.util.is_shape(a, (-1, (4, 5)))
    Out[7]: False
    """

    # if the obj.shape is different length than
    # the goal shape it means they have different number
    # of dimensions and thus the obj is not the query shape
    if (not hasattr(obj, 'shape') or
            len(obj.shape) != len(shape)):
        return False

    # empty lists with any flexible dimensions match
    if len(obj) == 0 and -1 in shape:
        return True

    # loop through each integer of the two shapes
    # multiple values are sequences
    # wildcards are less than zero (i.e. -1)
    for i, target in zip(obj.shape, shape):
        # check if current field has multiple acceptable values
        if is_sequence(target):
            if i in target:
                # obj shape is in the accepted values
                continue
            else:
                return False

        # check if current field is a wildcard
        if target < 0:
            if i == 0 and not allow_zeros:
                # if a dimension is 0, we don't allow
                # that to match to a wildcard
                # it would have to be explicitly called out as 0
                return False
            else:
                continue
        # since we have a single target and a single value,
        # if they are not equal we have an answer
        if target != i:
            return False

    # since none of the checks failed the obj.shape
    # matches the pattern
    return True

def discrete_gaussian_curvature_measure(mesh, points, radius):
    """
    Return the discrete gaussian curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.
    This is the sum of the vertex defects at all vertices
    within the radius for each point.
    Parameters
    ----------
    points : (n, 3) float
      Points in space
    radius : float ,
      The sphere radius, which can be zero if vertices
      passed are points.
    Returns
    --------
    gaussian_curvature:  (n,) float
      Discrete gaussian curvature measure.
    """

    points = np.asanyarray(points, dtype=np.float64)
    if not is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    nearest = mesh.kdtree.query_ball_point(points, radius)
    gauss_curv = [mesh.vertex_defects[vertices].sum() for vertices in nearest]

    return np.asarray(gauss_curv)

def line_ball_intersection(start_points, end_points, center, radius):
    """
    Compute the length of the intersection of a line segment with a ball.
    Parameters
    ----------
    start_points : (n,3) float, list of points in space
    end_points   : (n,3) float, list of points in space
    center       : (3,) float, the sphere center
    radius       : float, the sphere radius
    Returns
    --------
    lengths: (n,) float, the lengths.
    """

    # We solve for the intersection of |x-c|**2 = r**2 and
    # x = o + dL. This yields
    # d = (-l.(o-c) +- sqrt[ l.(o-c)**2 - l.l((o-c).(o-c) - r^**2) ]) / l.l
    L = end_points - start_points
    oc = start_points - center  # o-c
    r = radius
    ldotl = np.einsum('ij, ij->i', L, L)  # l.l
    ldotoc = np.einsum('ij, ij->i', L, oc)  # l.(o-c)
    ocdotoc = np.einsum('ij, ij->i', oc, oc)  # (o-c).(o-c)
    discrims = ldotoc**2 - ldotl * (ocdotoc - r**2)

    # If discriminant is non-positive, then we have zero length
    lengths = np.zeros(len(start_points))
    # Otherwise we solve for the solns with d2 > d1.
    m = discrims > 0  # mask
    d1 = (-ldotoc[m] - np.sqrt(discrims[m])) / ldotl[m]
    d2 = (-ldotoc[m] + np.sqrt(discrims[m])) / ldotl[m]

    # Line segment means we have 0 <= d <= 1
    d1 = np.clip(d1, 0, 1)
    d2 = np.clip(d2, 0, 1)

    # Length is |o + d2 l - o + d1 l|  = (d2 - d1) |l|
    lengths[m] = (d2 - d1) * np.sqrt(ldotl[m])

    return lengths


def sphere_ball_intersection(R, r):
    """
    Compute the surface area of the intersection of sphere of radius R centered
    at (0, 0, 0) with a ball of radius r centered at (R, 0, 0).
    Parameters
    ----------
    R : float, sphere radius
    r : float, ball radius
    Returns
    --------
    area: float, the surface are.
    """
    x = (2 * R**2 - r**2) / (2 * R)  # x coord of plane
    if x >= -R:
        return 2 * np.pi * R * (R - x)
    if x < -R:
        return 4 * np.pi * R**2

def discrete_mean_curvature_measure(mesh, points, radius):
    """
    Return the discrete mean curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.
    This is the sum of the angle at all edges contained in the
    sphere for each point.
    Parameters
    ----------
    points : (n, 3) float
      Points in space
    radius : float
      Sphere radius which should typically be greater than zero
    Returns
    --------
    mean_curvature : (n,) float
      Discrete mean curvature measure.
    """

    points = np.asanyarray(points, dtype=np.float64)
    if not is_shape(points, (-1, 3)):
        raise ValueError('points must be (n,3)!')

    # axis aligned bounds
    bounds = np.column_stack((points - radius,
                              points + radius))

    # line segments that intersect axis aligned bounding box
    candidates = [list(mesh.face_adjacency_tree.intersection(b))
                  for b in bounds]

    mean_curv = np.empty(len(points))
    for i, (x, x_candidates) in enumerate(zip(points, candidates)):
        endpoints = mesh.vertices[mesh.face_adjacency_edges[x_candidates]]
        lengths = line_ball_intersection(
            endpoints[:, 0],
            endpoints[:, 1],
            center=x,
            radius=radius)
        angles = mesh.face_adjacency_angles[x_candidates]
        signs = np.where(mesh.face_adjacency_convex[x_candidates], 1, -1)
        mean_curv[i] = (lengths * angles * signs).sum() / 2

    return mean_curv

def standardize(data, f, mean_std):
    data_mean = mean_std[f][0]  
    data_std = mean_std[f][1]  
    normalized_data = (data - data_mean)/data_std
    f +=1
    return normalized_data, f

def expand_df(df):
    d1 = {'teams': [['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],
                ['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG'],['SF', 'NYG']]}
    df2 = pd.DataFrame(df)
    # A3,D1,D2,D3,D4
    curvature = pd.DataFrame(df['curvature'].to_list(), columns=['curvature_1','curvature_2',"curvature_3","curvature_4","curvature_5","curvature_6","curvature_7","curvature_8"])
    A3 = pd.DataFrame(df['A3'].to_list(), columns=['A3_1','A3_2',"A3_3","A3_4","A3_5","A3_6","A3_7","A3_8"])
    D1 = pd.DataFrame(df['D1'].to_list(), columns=['D1_1','D1_2',"D1_3","D1_4","D1_5","D1_6","D1_7","D1_8"])
    D2 = pd.DataFrame(df['D2'].to_list(), columns=['D2_1','D2_2',"D2_3","D2_4","D2_5","D2_6","D2_7","D2_8"])
    D3 = pd.DataFrame(df['D3'].to_list(), columns=['D3_1','D3_2',"D3_3","D3_4","D3_5","D3_6","D3_7","D3_8"])
    D4 = pd.DataFrame(df['D4'].to_list(), columns=['D4_1','D4_2',"D4_3","D4_4","D4_5","D4_6","D4_7","D4_8"])

    df_scalar = df[df.columns[1:8]]
    # df_scalar = df_scalar.conca (curvature,left_on='curvature',right_on="curvature")
    curv_list = ['curvature_1','curvature_2',"curvature_3","curvature_4","curvature_5","curvature_6","curvature_7","curvature_8"]
    a3_list = ['A3_1','A3_2',"A3_3","A3_4","A3_5","A3_6","A3_7","A3_8"]
    d1_list = ['D1_1','D1_2',"D1_3","D1_4","D1_5","D1_6","D1_7","D1_8"]
    d2_list = ['D2_1','D2_2',"D2_3","D2_4","D2_5","D2_6","D2_7","D2_8"]
    d3_list = ['D3_1','D3_2',"D3_3","D3_4","D3_5","D3_6","D3_7","D3_8"]
    d4_list = ['D4_1','D4_2',"D4_3","D4_4","D4_5","D4_6","D4_7","D4_8"]

    for column in curv_list:
        df_scalar[column] = curvature[column]
    for column in a3_list:
        df_scalar[column] = A3[column]
    for column in d1_list:
        df_scalar[column] = D1[column]
    for column in d2_list:
        df_scalar[column] = D2[column] 
    for column in d3_list:
        df_scalar[column] = D3[column]
    for column in d4_list:
        df_scalar[column] = D4[column] 
    
    return df_scalar

def predict(normalized_data, query, sampled_min_max, sampled_mean_std, n):
    comaprsion_df = normalized_data.copy()
    scalar_query_features = np.array(query[1:8], dtype = 'float32')
    descriptors_query_features = np.vstack(np.array(query[8:]))
    comaprsion_values = {}
    for index, row in normalized_data.iterrows():
        scalar_features = np.array(row[1:8], dtype = 'float32')
        descriptor_features = np.vstack(np.array(row[8:]))
        scalar_distance = distance.cosine(scalar_query_features,scalar_features)
        total_descriptor_distance = 0
        for i in range(len(descriptors_query_features)):
            descriptor_query_feature = descriptors_query_features[i]
            descriptor_feature = descriptor_features[i]
            descriptor_distance = wasserstein_distance(descriptor_query_feature,descriptor_feature)

            descriptor_distance_scaled = (descriptor_distance - sampled_min_max[i][0]) / (sampled_min_max[i][1] - sampled_min_max[i][0])
            total_descriptor_distance += descriptor_distance
            total_descriptor_distance = total_descriptor_distance
        total_distance = scalar_distance + total_descriptor_distance

        if np.isnan(total_distance ): 
            comaprsion_values[index] = 999
        else:
            comaprsion_values[index] = total_distance
    
    extract = lambda x: x[0]
    pred = list(map(extract,sorted(comaprsion_values.items(), key=lambda x: x[1])[:n]))
    return pred

def get_query(pred_list):
    DB_DIRECTORY = r"full_normalized_benchmark\**\*.off"
    mesh_files = list(glob.glob(DB_DIRECTORY,recursive=True))
    mesh_files.sort(key=natural_keys)

    for j1 in pred_list:
        for j,mesh_file in enumerate(mesh_files):
        	if j==j1:
        		mesh = trimesh.load(mesh_file,force='mesh')
        		view_mesh(mesh)