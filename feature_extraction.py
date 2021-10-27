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

df = load_dataset()
mesh_files = dir_to_sorted_file_list()
SAVE_PATH = "database/norm_db_waterproof.csv"
LOAD_PATH="database/normalized_db.csv"

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

def stats_to_fig(data,column_stat):  
    '''
    calculates statistics for dataframe
    params: amnt_vertices, amnt_faces
    '''      
    
    # df = df[df["amnt_faces"]<3500]
    # data = df[param]
    sorted_vertices = np.sort(data)
    bins = np.arange(np.max(data))[::5000]  #steps on x axis
    # bins = np.linspace(0,1,11) # N / amount per N = bins
    print(np.max(data))

    fig,ax = plt.subplots(figsize=(15,7))
    ax.hist(sorted_vertices,bins=bins)
    ax.set_xticks(bins)
    # ax.set
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    

    # set title and labels
    ax.set_title(column_stat,fontsize=20)
    ax.set_xlabel(f"distance between barycenter and origin",fontsize=20)
    ax.set_ylabel("count meshes",fontsize=20)

    # save figure
    plt.savefig(f"stats/feature_extraction/{column_stat}")

if __name__=="__main__":
    #before this also rescale mesh with vertices and faces then we can normalize a query object towards same as database objects!
    features_df = pd.DataFrame(columns = ['id','surface_area', 'compactness','sphericity','volume','diameter','rectangulairty','eccentricity','curvature', 'A3', 'D1', 'D2', 'D3', 'D4'])
    
    fig, axs = plt.subplots(5, figsize=(5,10))
    for i,mesh_file in enumerate(mesh_files):
    	print(i, end='\r')  
    	features = []
    	if(i !=1000000):
         
            mesh = trimesh.load(mesh_file,force='mesh')
            #view_mesh(mesh)
            
            #fixed = sk.pre.fix_mesh(mesh, remove_disconnected=0, inplace=False)
            #skel = sk.skeletonize.by_wavefront(mesh, waves=3, step_size=1)
            #skel.save_swc('skeleton.swc')
            #print(skel.vertices)
            #skel_pc = PointCloud(skel.vertices)
            #print(skel_pc.centroid)
            #skel_pc.show()
            #
            # if not mesh.is_watertight:
            #     view_mesh(mesh)
            
            # VOLUME
            #print(mesh.is_watertight)

            #PointCloud(mesh.vertices, colors=None).show()S
            # print(points)
            #view_mesh(mesh)

            data = discrete_gaussian_curvature_measure(mesh,mesh.vertices, 0.1)
            
            #data = discrete_mean_curvature_measure(mesh,mesh.vertices,0.1)
            # print(min(data),max(data))
            local_weight = 0
            global_weight = 1
            scaler = MinMaxScaler()
            norm_data = np.array(local_weight*data - global_weight*data).reshape(-1,1)
            #print(norm_data.shape)
            norm_data = scaler.fit_transform(norm_data)
            #print(norm_data)
            norm_hist, _ = np.histogram(norm_data,bins=8)
            curvature = norm_hist

            norm = matplotlib.colors.Normalize(vmin=(local_weight*min(data))-(global_weight*40), vmax=(local_weight*max(data))+ (global_weight*30), clip=True)
            #norm = matplotlib.colors.Normalize(vmin=1.0*min(data)-(40*1.0), vmax=1.0*max(data)+ (30*1.0), clip=True)
            #norm2 = matplotlib.colors.Normalize(vmin=-30, vmax=40, clip=True)
       		
            mapper = cm.ScalarMappable(norm=norm, cmap=cm.turbo)

            node_color = [(r, g, b) for r, g, b, a in mapper.to_rgba(data)]
            #print(len(node_color))

            # to view point cloud
            #PointCloud(mesh.vertices, colors=node_color).show()

            #print(discrete_gaussian_curvature_measure(mesh,mesh.vertices, 0.001))
           # print("Vertices",len(mesh.vertices))
            #print(len(discrete_gaussian_curvature_measure(mesh,mesh.vertices, 0.001)))
            #print(discrete_mean_curvature_measure(mesh,mesh.vertices, 0.001))
            
            pc_mesh = PointCloud(mesh.vertices).convex_hull

            diameter = np.max(pc_mesh.bounds[1]-pc_mesh.bounds[0])
            #diameter = np.sum((pc_mesh.bounds[1]-pc_mesh.bounds[0])**2)
            #print(diameter)
            #print("Watertight",pc_mesh.is_watertight)
             
            # AREA
            #print("Area",mesh.area)
            features.append(i)
            features.append(mesh.area)

            # Compactness
            compactness = (pc_mesh.area**3)/ (36* np.pi*(pc_mesh.volume**2))
            sphericity = 1/compactness
            features.append(compactness)
            features.append(sphericity)
            #print("compactness",compactness)
            #print("sphericity",sphericity)

            
            # RECTENGULARITY  
            volume =   pc_mesh.volume        
            #print("boundbox_volume",volume)
            features.append(volume)

            features.append(diameter)
            rectangularity = pc_mesh.volume / mesh.bounding_box.volume
            features.append(rectangularity)
            #print("Rectangularity", rectangularity)
                  

            # Eccentricity 
            eccentricity  = abs(mesh.principal_inertia_components[0] / mesh.principal_inertia_components[2])
            #eccentricity  = mesh.principal_inertia_vectors[0]/mesh.principal_inertia_components[2]
            features.append(eccentricity)
            
            #print(curvature)
            features.append(curvature)
            #print("Eccentricity", eccentricity)
            
            A3 = []
            D1 = []
            D2 = []
            D3 = []
            D4 = []
            # A3 angle between 3 random vertices
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
                # 	print(norm_v1, norm_v2)
                # 	print("Division by 0")
                if math.isnan(np.dot(vec1,vec2) / (norm_v1* norm_v2) ):
                	j-=1
                	continue
                angle = (np.rad2deg(np.arccos( np.dot(vec1,vec2) / (norm_v1* norm_v2) )))
                if math.isnan(angle):
                	j-=1
                	continue
                A3.append(angle)

                #print("Angle", angle)

                # D1 Distance
                distance = np.linalg.norm(v1-v_bary)
                D1.append(distance)
                #print("D1", distance)
                # D2 distance
                distance_2 = np.linalg.norm(v1-v2)
                D2.append(distance_2)
                #print("D2", distance_2)

                # Area 3 vertices
                crosses = np.array([np.cross(vec1,vec2)])
                area = (np.sum(crosses**2, axis=1)**.5) * .5
                distance_3 = min(np.sum(area), 0.1)
                D3.append(distance_3)
                #print("D3", distance_3)      
                # print(i,distance_3)   
                # print(i,np.linalg.norm(crosses)/2)
                # print(i,np.sqrt(np.dot(np.cross(vec1,vec2),np.cross(vec1,vec2).T))/2)

                # cube root of volume formed by tetrahedron of 4 random vertices
                prod = np.linalg.norm( np.dot(np.cross(vec1,vec2),vec3))
                volume = (1/6)*prod
                cube_root = volume**(1/3)
                D4.append(cube_root)
                #print("D4", cube_root)

            #print(D1)
            A3_descriptor, x = np.histogram(A3,bins=8)
            # bin_centers = np.arange(1,9)
            # axs[0].plot(bin_centers, A3_descriptor)
            # axs[0].set_title("A3 shape descriptor")
            # axs[0].set_xticks(np.arange(1,9))
            # axs[0].set_yticks(np.arange(0,16000,5000))


            D1_descriptor, x = np.histogram(D1,bins=8)
            # bin_centers = np.arange(1,9)
            # axs[1].plot(bin_centers, D1_descriptor)
            # axs[1].set_title("D1 shape descriptor")
            # axs[1].set_xticks(np.arange(1,9))
            # axs[1].set_yticks(np.arange(0,31000,10000))

            D2_descriptor, x = np.histogram(D2,bins=8)
            # bin_centers = np.arange(1,9)
            # axs[2].plot(bin_centers, D2_descriptor)
            # axs[2].set_title("D2 shape descriptor")
            # axs[2].set_xticks(np.arange(1,9))
            # axs[2].set_yticks(np.arange(0,31000,10000))

            D3_descriptor, x = np.histogram(D3,bins=8)
            # bin_centers = np.arange(1,9)
            # axs[3].plot(bin_centers, D3_descriptor)
            # axs[3].set_title("D3 shape descriptor")
            # axs[3].set_yticks(np.arange(0,31000,10000))

            D4_descriptor, x = np.histogram(D4,bins=8)
            # bin_centers = np.arange(1,9)
            # axs[4].plot(bin_centers, D4_descriptor)
            # axs[4].set_title("D4 shape descriptor")
            # axs[4].set_xticks(np.arange(1,9))
            # axs[4].set_yticks(np.arange(0,31000,10000))

            features.append(A3_descriptor)
            features.append(D1_descriptor)
            features.append(D2_descriptor)
            features.append(D3_descriptor)
            features.append(D4_descriptor)
            # print(len(features))
            # print(len(features_df.columns))
            features_df.loc[len(features_df)] = features
            features_df.to_csv (r'features_final_df.csv', index = False, header=True)
            fig.tight_layout(pad=1.2)
            # #print(i)
            # if(i==1310):
            # 	plt.savefig('helicopter_descriptors.png')
            # 	plt.show()
            #features_df = features_df.append(pd.DataFrame(features))
            #print(A3_descriptor, D1_descriptor,D2_descriptor, D3_descriptor, D4_descriptor)
             
                # print( cube_root)
                    
            #stats_to_fig(angle,"angle between 3 vertices")
            #break

               
            # view_mesh(mesh)

            # D1 angle between bary center and random vertex
            # break


            # if not mesh.is_watertight:
            #     print(len(mesh.faces))
            #     print("before",mesh.is_watertight)
            #     fill_holes(mesh)
            #     print("before",mesh.is_watertight)
            #     view_mesh(mesh)
            #     break

            
