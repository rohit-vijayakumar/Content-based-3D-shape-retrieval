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

from project_statistics import get_outliers
from create_dataset import load_dataset,dir_to_sorted_file_list

df = load_dataset()
mesh_files = dir_to_sorted_file_list()
SAVE_PATH = "database/norm_db_waterproof.csv"
LOAD_PATH="database/normalized_db.csv"

def operate_mesh(index,subdivide=0,decimate=0,inp_mesh=None):
    # load mesh from og database 
    mesh  = trimesh.load(mesh_files[index],force='mesh')
    # load mesh from subdivided input mesh
    if not type(inp_mesh) == type(None):
        mesh = inp_mesh 

    # use subdivision or decimation
    for i in range(subdivide):
        remesh.subdivide2(mesh)    
    for i in range(decimate):
        mesh = remesh.simplify_quadratic_decimation(mesh,decimate,Trimesh=Trimesh)
    return mesh


def view_mesh(mesh):        
    pyrend_mesh = pyrender.Mesh.from_trimesh(mesh)
    box = pyrender.Mesh.from_trimesh(mesh.bounding_box_oriented)
    
    scene = pyrender.Scene()
    scene.add(pyrend_mesh)
    scene.add(box)
    pyrender.Viewer(scene, use_raymond_lighting=True)

# goal range vertices between 700 and 2450
def generate_normalized_db():
    df = pd.DataFrame(columns=["id","water_tight"])
    for i,mesh_file in enumerate(mesh_files):    
 
        save_folder = '/'.join(mesh_file.replace("benchmark","normalized_benchmark").split('\\')[:4])
        save_name = '/'.join(mesh_file.replace("benchmark","normalized_benchmark").split('\\')[4:])
        try: os.makedirs(save_folder) 
        except: pass #file already exists

        mesh = trimesh.load(mesh_file,force='mesh')
        amnt_vertices = len(mesh.vertices)
        amnt_faces = len(mesh.faces)
        if amnt_vertices < 900: # vertices * 3.5 = vertices after 1 subdivide 
            n = round(math.log(1000/amnt_vertices,3))
            mesh = operate_mesh(index=i,subdivide=n)
            amnt_vertices = len(mesh.vertices)
            amnt_faces = len(mesh.faces)

        if amnt_vertices > 1250:
            # values now range between 850 vertices and 1250 because of decimation 
            mesh=operate_mesh(index=i,decimate=1850,inp_mesh=mesh) #vertices = approximately  faces/1.85
        df = df.append({"id":i,"water_tight":int(mesh.is_watertight)},ignore_index=True)
        
        # view_mesh(mesh)
        mesh.export(f"{save_folder}/{save_name}",file_type="off")
        
        if i%50 == 0 and i >0: 
            print(i)
         
    df.to_csv(SAVE_PATH,index=False)

import open3d as o3d
import copy
# if __name__=="__main__":
#     pass
#     # generate_normalized_db() # already done
#     for i,mesh_file in enumerate(mesh_files):    
#         if  i == 1123:
#             mesh = trimesh.load(mesh_file,force='mesh')
#             print(mesh.center_mass)

    

#             mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
#             # mesh_tx = copy.deepcopy(mesh).translate((-0.05167548, -0.05167548, -0.05167548))
#             mesh_tx  = copy.deepcopy(mesh).create_coordinate_frame(size=1.0,origin=np.array([-0.1499,-0.1499,-0.1499]))
           
#             print(f'Center of mesh: {mesh.get_center()}')
#             print(f'Center of mesh tx: {mesh_tx.get_center()}')
#             mesh = Trimesh(vertices=mesh_tx.vertices, faces=mesh_tx.triangles)
#             print(mesh.center_mass)
            # print(mesh.triangles)#.points_to_barycentric())
            # o3d.visualization.draw_geometries([mesh, mesh_tx])

            # mesh = Trimesh(mesh)
            # print(mesh.center_mass)
    


         


''' view outliers
if __name__=="__main__":
    pass
    # generate_normalized_db() # already done
    for i,mesh_file in enumerate(mesh_files):    

       
        if  i == 1648 or i == 1736:
            mesh = trimesh.load(mesh_file,force='mesh')
     
            amnt_vertices = len(mesh.vertices)
            amnt_faces = len(mesh.faces)
            print(amnt_vertices)
            print(amnt_faces)
            if amnt_vertices < 900: # vertices * 3.5 = vertices after 1 subdivide 
                n = round(math.log(1000/amnt_vertices,3))
                mesh = operate_mesh(index=i,subdivide=n)
                amnt_vertices = len(mesh.vertices)
                amnt_faces = len(mesh.faces)
      

            if amnt_vertices > 1250:
                # values now range between 850 vertices and 1250 because of decimation 
                mesh=operate_mesh(index=i,decimate=1850,inp_mesh=mesh) #vertices = approximately  faces/1.85
            amnt_vertices = len(mesh.vertices)
            amnt_faces = len(mesh.faces)
            print(amnt_vertices)
            print(amnt_faces)
            view_mesh(mesh)

'''