from matplotlib.pyplot import draw, fill
import pyrender
import trimesh
from trimesh import remesh
import pandas as pd
import numpy as np
from trimesh import Trimesh
import matplotlib.pyplot as plt
import math
import os
from view_data import view_mesh
import copy

from project_statistics import get_outliers
from create_dataset import load_dataset,dir_to_sorted_file_list
from trimesh.repair import fill_holes
from trimesh.points import PointCloud

import random

df = load_dataset()
mesh_files = dir_to_sorted_file_list()
SAVE_PATH = "database/norm_db_waterproof.csv"
LOAD_PATH="database/normalized_db.csv"

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
    for i,mesh_file in enumerate(mesh_files):   
        if  i >= 0 and i <20 :
         
            mesh = trimesh.load(mesh_file,force='mesh') 
            if not mesh.is_watertight:
                view_mesh(mesh)
            '''
            # VOLUME
            print(mesh.is_watertight)
            pc_mesh = PointCloud(mesh.vertices).convex_hull
            print(pc_mesh.is_watertight)
               
            # AREA
            print(mesh.area)

            print("pi",np.pi)
            # Compactness
            compactness = (pc_mesh.area**3)/ (36* np.pi*(pc_mesh.volume**2))
            sphericity = 1/compactness
            print("compactness",compactness)
            print("sphericity",sphericity)

            
            # RECTENGULARITY            
            print("boundbox_volume",mesh.bounding_box.volume)
            print(pc_mesh.volume / mesh.bounding_box.volume)
            '''      

            # Essentricity
            # essentricity = mesh.principal_inertia_vectors[0]/mesh.principal_inertia_components[2]
            
            # A3 angle between 3 random vertices
            v_bary = mesh.centroid
            for i in range (50000):
                v1 = mesh.vertices[random.randint(0,len(mesh.vertices)-1)] 
                v2 = mesh.vertices[random.randint(0,len(mesh.vertices)-1)] 
                v3 = mesh.vertices[random.randint(0,len(mesh.vertices)-1)]
                v4 = mesh.vertices[random.randint(0,len(mesh.vertices)-1)]
                # v1 = np.array([1,-4,-2])
                # v2 = np.array([3,-3,-3])
                # v3 = np.array([5,-1,-2])

                v1 = np.array([-1,2,0])
                v2 = np.array([2,1,-3])
                v3 = np.array([1,0,1])
                v4 = np.array([3,-2,3])
                
                # A3 Angle
                vec1 = v2-v1
                vec2 = v3-v1
                vec3 = v4-v1

                norm_v1 = np.linalg.norm(vec1)
                norm_v2 = np.linalg.norm(vec2)
            
                angle = (np.rad2deg( np.arccos( np.dot(vec1,vec2) / (norm_v1* norm_v2) )))
            
                # D1 Distance
                distance = v1-v_bary

                # D2 distance
                distance_2 = v1-v2

                # Area 3 vertices
                crosses = np.array([np.cross(vec1,vec2)])
                area = (np.sum(crosses**2, axis=1)**.5) * .5
                distance_3 = np.sum(area)        
                # print(i,distance_3)   
                # print(i,np.linalg.norm(crosses)/2)
                # print(i,np.sqrt(np.dot(np.cross(vec1,vec2),np.cross(vec1,vec2).T))/2)

                # cube root of volume formed by tetrahedron of 4 random vertices
                prod = np.linalg.norm( np.dot(np.cross(vec1,vec2),vec3))
                volume = (1/6)*prod
                cube_root = volume**(1/3)
                # print( cube_root)
                    
            stats_to_fig(angle,"angle between 3 vertices")
            break

               
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

            
