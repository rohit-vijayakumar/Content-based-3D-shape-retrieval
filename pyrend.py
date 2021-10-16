from matplotlib.pyplot import draw
import pyrender
import trimesh
from trimesh import remesh
import pandas as pd
import numpy as np
from trimesh import Trimesh
import matplotlib.pyplot as plt

from create_dataset import load_dataset,dir_to_sorted_file_list

df = load_dataset()
mesh_files = dir_to_sorted_file_list()

def show_stats(df):    
    data = df["amnt_vertices"]
    sorted_vertices = np.sort(data)
    bins = np.arange(np.max(data))[::1000]
    fig,ax = plt.subplots(figsize=(15,7))
    ax.hist(sorted_vertices,bins=bins)
    ax.set_xticks(bins)
    plt.savefig("stats/vertices")
    plt.show()

show_stats(df)
print(5/0)
print(df["amnt_vertices"].describe())
print(df[df["amnt_vertices"]<50]["id"])


for i,mesh_file in enumerate(mesh_files):
    id = i
    class_id = mesh_file.split("db")[1].split("\\")[1]
    
    # load mesh
    mesh = trimesh.load(mesh_file,force='mesh')

    print(5/0)
    # mesh = remesh.subdivide2(mesh)

    # mesh = remesh.subdivide2(mesh)
    if i == 56:
        print(len(mesh.faces))
        for q in range(3):
            remesh.subdivide2(mesh)
            # r = remesh.simplify_quadratic_decimation(mesh,1000,Trimesh)
            # print(len(r.triangles))
        mesh_f = pyrender.Mesh.from_trimesh(mesh)
        
        scene = pyrender.Scene()
        scene.add(mesh_f)
        pyrender.Viewer(scene, use_raymond_lighting=True)
        break
        # mesh.show()
    # calculate mesh statistics
    # amnt_vertices =  len(mesh.vertices)
    # amnt_faces    =  len(mesh.faces)    
    # is_triangle   =  int(mesh.faces.shape[1]) == 3
    # bounding_box  =   mesh.bounds.flatten()
    # df = df.append({"id":id,"class_id":class_id,"amnt_vertices":amnt_vertices,"amnt_faces":amnt_faces,"is_triangle":is_triangle,"bounding_box":bounding_box},ignore_index=True)
    # if i == 100:
    #     break

# df = pd.DataFrame(columns=["id","class_id","amnt_vertices","amnt_faces",])
# print(df)
# for mesh in enumerate(mesh_files):
#     print(mesh)
# mesh = trimesh.load(mesh_files[0],force='mesh')
# # print(mesh.faces.shape)

# for i,mesh_file in enumerate(mesh_files):    
#     if i <10 and i >0:
#         mesh = trimesh.load(mesh_file,force='mesh')
#         # print(mesh.faces.shape)
#         # mesh = mesh + mesh.bounding_box_oriented
#         mesh_f = pyrender.Mesh.from_trimesh(mesh)
#         box = pyrender.Mesh.from_trimesh(mesh.bounding_box_oriented)
#         scene = pyrender.Scene()
#         scene.add(mesh_f)
#         scene.add(box)
#         # scene.add(mesh.bounding_box_orieneted)
#         pyrender.Viewer(scene, use_raymond_lighting=True)
#     else:
#         pass
# #     vertices.append(len(mesh.vertices))
# # print(len(mesh.faces))


# print(vertices[:10])
# mesh = fuze_trimesh

# print(len(mesh.vertices))
# print(len(a))
# mesh = trimesh.load("Shadowboy_Idle.obj",force='mesh')







# #apperently works in place
# # mesh = remesh.subdivide2(fuze_trimesh)
# # mesh = remesh.subdivide2(fuze_trimesh)
# # mesh = remesh.subdivide2(fuze_trimesh)
# # mesh = remesh.subdivide2(fuze_trimesh)

# mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)



# scene = pyrender.Scene()
# scene.add(mesh)
# pyrender.Viewer(scene, use_raymond_lighting=True)