from matplotlib.pyplot import draw
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
    # bins = np.arange(np.max(data))[::1] /100 #steps on x axis
    bins = np.linspace(-1,1,11) # N / amount per N = bins
    # print(np.max(data))

    fig,ax = plt.subplots(figsize=(15,7))
    ax.hist(sorted_vertices,bins=bins)
    ax.set_xticks(bins)
    # ax.set
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    

    # set title and labels
    ax.set_title(column_stat,fontsize=20)
    ax.set_xlabel(f"Dot product between major eigenvector and X axis",fontsize=20)
    ax.set_ylabel("count meshes",fontsize=20)

    # save figure
    plt.savefig(f"stats/{column_stat}")
def stats_to_fig2(data,column_stat):  
    '''
    calculates statistics for dataframe
    params: amnt_vertices, amnt_faces
    '''      
    
    # df = df[df["amnt_faces"]<3500]
    # data = df[param]
    sorted_vertices = np.sort(data)
    change = lambda x: 1 if x else 0
    sorted_vertices = list(map(change,sorted_vertices))
    # print(sorted_vertices)
    # bins = np.arange(np.max(data))[::1] /100 #steps on x axis
    bins = np.linspace(0,1,2) # N / amount per N = bins
    # print(np.max(data))

    fig,ax = plt.subplots(figsize=(15,7))
    ax.bar(["wrong flip","correct flip"],(sorted_vertices.count(1),sorted_vertices.count(0)))
    # ax.set_xticks(bins)
    # ax.set
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    

    # set title and labels
    ax.set_title(column_stat,fontsize=20)
    ax.set_xlabel(f"flip state of mesh in all directions",fontsize=20)
    ax.set_ylabel("count meshes",fontsize=20)

    # save figure
    plt.savefig(f"stats/{column_stat}")

def stats_to_fig3(data,column_stat):  
    '''
    calculates statistics for dataframe
    params: amnt_vertices, amnt_faces
    '''      
    
    # df = df[df["amnt_faces"]<3500]
    # data = df[param]
    sorted_vertices = np.sort(data)
    change = lambda x: 1 if x else 0
    sorted_vertices = list(map(change,sorted_vertices))
    # print(sorted_vertices)
    # bins = np.arange(np.max(data))[::1] /100 #steps on x axis
    bins = np.linspace(0,1,2) # N / amount per N = bins
    # print(np.max(data))

    fig,ax = plt.subplots(figsize=(15,7))
    ax.bar(["Wrong size","Correct size"],(sorted_vertices.count(0),sorted_vertices.count(1)))
    # ax.set_xticks(bins)
    # ax.set
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    

    # set title and labels
    ax.set_title(column_stat,fontsize=20)
    ax.set_xlabel(f"amount of times a mesh was scaled to fit in unit cube",fontsize=20)
    ax.set_ylabel("count meshes",fontsize=20)

    # save figure
    plt.savefig(f"stats/{column_stat}")
if __name__=="__main__":
    # generate_normalized_db() # already done
    pre_bary_norm = []
    post_bary_norm = []
    pre_scale = []
    post_scale = []
    t = []
    pre_angle_eigenvector = []
    post_angle_eigenvector = []
    pre_flip_hist = []
    post_flip_hist = []
    #before this also rescale mesh with vertices and faces then we can normalize a query object towards same as database objects!
    for i,mesh_file in enumerate(mesh_files):   
        if  i >= 0 and i < 10000 :
            save_folder = '/'.join(mesh_file.replace("normalized2_benchmark","full_normalized_benchmark").split('\\')[:4])
            save_name = '/'.join(mesh_file.replace("normalized2_ benchmark","full_normalized_benchmark").split('\\')[4:])
            try: os.makedirs(save_folder) 
            except: pass #file already exists

            mesh = trimesh.load(mesh_file,force='mesh') 
        
            
            # normalize centroid (stats)        
            # pre normalize statistics    
            if np.mean(abs(mesh.centroid) < 1): pre_mass = np.mean(abs(mesh.centroid))  
            else: t.append(np.mean(abs(mesh.centroid)))
            pre_bary_norm.append(pre_mass)
            # centroid translation
            mesh.apply_translation(-mesh.centroid)    
            # post normalize statistics 
            post_mass = np.mean(abs(mesh.centroid)) 
            post_bary_norm.append(post_mass)
            # normalize rotaton orientation
            pre_angle_eigenvector.append(np.dot(mesh.principal_inertia_vectors[0],np.array([1,0,0]) ))
            mesh.apply_transform(mesh.principal_inertia_transform)
            post_angle_eigenvector.append(np.dot(mesh.principal_inertia_vectors[0],np.array([1,0,0]) ))
            # print(np.dot(mesh.principal_inertia_vectors[0],np.array([1,0,0])))
            # print((mesh.integral_mean_curvature))
            # mesh.bounding_box_oriented
            # view_mesh(mesh)
            # print(mesh.area)

            # moment test flip 
            bool_mask = np.zeros(3)
            a = 0
            b = 0
            top1 = 0
            top2 = 0
            left1 = 0
            left2 = 0
            front1 = 0
            front2 = 0
            for j in range(2): # what principal axis to cut
                mesh_p1 = copy.deepcopy(mesh.slice_plane(mesh.centroid,mesh.principal_inertia_vectors[0],flip=j)) # should use this for pca
                mesh_p2 = copy.deepcopy(mesh.slice_plane(mesh.centroid,mesh.principal_inertia_vectors[1],flip=j)) 
                mesh_p3 = copy.deepcopy(mesh.slice_plane(mesh.centroid,mesh.principal_inertia_vectors[2],flip=j)) 
                # print(mesh.principal_inertia_vectors[0])
                # print(mesh.principal_inertia_vectors[1])
                # print("centroid",mesh.centroid,"center mass",mesh.center_mass)
                # if i == 0: view_mesh(mesh_p1)
                # if i == 1: view_mesh(mesh_p1)
                # view_mesh(p2)
                mesh_p1_area = mesh_p1.area; bool_mask[0] = mesh_p1_area if j ==0 else (mesh_p1_area > bool_mask[0]) # 1 = bot heavier than top
                mesh_p2_area = mesh_p2.area; bool_mask[1] = mesh_p2_area if j ==0 else (mesh_p2_area > bool_mask[1]) # 1 = left heavier than right
                mesh_p3_area = mesh_p3.area; bool_mask[2] = mesh_p3_area if j ==0 else (mesh_p3_area > bool_mask[2]) # 1 = front (camera front view) heavier than back view

                z_axis = lambda x: x[2]
                y_axis = lambda x: x[1]
                x_axis = lambda x: x[0]
                if j == 0: 
                    top1 = np.mean(list(map(z_axis,mesh_p1.vertices)))
                    left1 = np.mean(list(map(y_axis,mesh_p2.vertices)))
                    front1= np.mean(list(map(x_axis,mesh_p3.vertices)))
                    # print("1st",top1)
                if j == 1: 
                    top2 = np.mean(list(map(z_axis,mesh_p1.vertices)))                    
                    left2 = np.mean(list(map(y_axis,mesh_p2.vertices)))                    
                    front2 =  np.mean(list(map(x_axis,mesh_p3.vertices)))
                    # print("scnd",top2)
       
                
                if j == 0: a = mesh_p1_area #/len(mesh_p1.vertices))
                if j ==1: b = mesh_p1_area
                # print(a,b)
                # bool_mask[0] = b>a
                # view_mesh(mesh_p1)
                # print(bool_mask)
            # print(bool_mask)
            # print(top2, top1)
            old_bool_mask = copy.deepcopy(bool_mask)
            if top2 > top1:
                # print("inverted z ax")
                for v,x in enumerate(bool_mask):
                    if x == 0:
                        bool_mask[v] =1
                    else:
                        bool_mask[v] = 0
                    break
            if left2 > left1:
                # print("inverted y ax" )
                for v,x in enumerate(bool_mask):
                    if v == 1:
                        if x == 0:
                            bool_mask[v] =1
                        else:
                            bool_mask[v] = 0
                        break
                    
            if front2 > front1:
                # print("inverted x ax" )
                # print(bool_mask)
                # view_mesh(mesh)
                for v,x in enumerate(bool_mask):

                    if v == 2:
                        if x == 0:
                            bool_mask[v] =1
                        else:
                            bool_mask[v] = 0
                        break
                # print(bool_mask)
          
            
            pre_flip_hist.append(False in np.equal(bool_mask,old_bool_mask)) #amnt of times not correctly alligned  = True, correctly alligned = False
            post_flip_hist.append(False)
            # view_mesh(mesh_p1)
            # print(5/0)
            # print(mesh.area) #able to calcualte temp area
            x_axis = lambda x: np.multiply(x,[-1,1,1]) # red
            y_axis = lambda x: np.multiply(x,[1,-1,1]) # green
            z_axis = lambda x: np.multiply(x,[1,1,-1]) # blue    
         
            if bool_mask[0]: mesh.vertices = np.array(list(map(z_axis,mesh.vertices))) # bot heavier than top => flip around z axis => top is heavier
            if bool_mask[1]: mesh.vertices = np.array(list(map(y_axis,mesh.vertices))) # left heavier than right => flip around y axis => right heavier than left            
            if not bool_mask[2]: mesh.vertices = np.array(list(map(x_axis,mesh.vertices))) # if front heavier than back => flip around x => move towards camera
                                                                                       #idea is that living things weight more on their back side thus
        
            # print(mesh.vertices[0])
            if front2>front1:
                # view_mesh(mesh)
                pass
            
            # view_mesh(mesh)
            # print(5/0)

            #normalize scale
     
            pre_scale.append( 0.998 <= max(mesh.bounds[1]-mesh.bounds[0]) <= 1.001  )
            mesh.apply_scale(1/max(mesh.bounds[1]-mesh.bounds[0])) # scale mesh to unit length
            post_scale.append(0.99998 <= max(mesh.bounds[1]-mesh.bounds[0]) <= 1.001 )
            # view_mesh(mesh)
            # mesh.export(f"{save_folder}/{save_name}",file_type="off")
            # print(5/0)

            # print(mesh.centroid)
            # print(bary_center_mesh,"boo")
            # print(5/0)


# print(np.mean(pre_bary_norm))
# print(np.mean(post_bary_norm),"avg barycenter")
print(sum(pre_scale))
print(sum(post_scale))
# print(len(t),np.mean(t))
# stats_to_fig(pre_bary_norm,"pre barycentric normalisation")
# stats_to_fig(post_bary_norm,"post barycentric normalisation")
stats_to_fig(pre_angle_eigenvector,"pre Pose normalisation")
stats_to_fig(post_angle_eigenvector,"post Pose normalisation")
# stats_to_fig2(pre_flip_hist,"pre Flip normalisation")
# stats_to_fig2(post_flip_hist,"post Flip normalisation")

# stats_to_fig3(pre_scale,"pre Scale normalisation")
# stats_to_fig3(post_scale,"post Scale normalisation")
# print(pre_flip_hist)
'''
def get_area_scaled_face_centroid(k,vertex_indices): # returns
        # oneliner for centroid 
        area_scaled_face_centroid = mesh.area_faces[k] * np.array((list(sum(mesh.vertices[j]/3 for j in (vertex_indices)))))
    
        avg = np.zeros(3)               
        for index in vertex_indices:
            avg += mesh.vertices[index]
        centroid = avg/3 #centroid of a face
        area_scaled_face_centroid = (centroid * mesh.area_faces[k])
    
        return area_scaled_face_centroid

    bary_center_mesh = np.zeros(3)
    for k,face in enumerate(mesh.faces):
        bary_center_mesh += get_area_scaled_face_centroid(k,face)
    bary_center_mesh /= mesh.area #-such that we translate in opposite direction
'''