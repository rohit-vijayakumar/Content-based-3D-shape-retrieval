import trimesh
import glob
import pandas as pd
import re

DB_DIRECTORY = r"normalized_benchmark\**\*.off"
SAVE_PATH = "database/raw_db.csv"
LOAD_PATH = "database/raw_db.csv"

def atoi(text):
    return int(text) if text.isdigit() else text

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def dir_to_sorted_file_list():
    mesh_files = list(glob.glob(DB_DIRECTORY,recursive=True))
    mesh_files.sort(key=natural_keys)
    return mesh_files

def create_data(mesh_files):
    df = pd.DataFrame(columns=["id","class_id","amnt_vertices","amnt_faces","water_tight"])
    for i,mesh_file in enumerate(mesh_files):        
        # load mesh
        mesh = trimesh.load(mesh_file,force='mesh')

        # calculate mesh statistics       
        id = i
        class_id      = mesh_file.split("db")[1].split("\\")[1]
        amnt_vertices =  len(mesh.vertices)
        amnt_faces    =  len(mesh.faces)    
        is_triangle   =  int(mesh.faces.shape[1]) == 3
        bounding_box  =  mesh.bounds.flatten()
        water_tight   =  int(mesh.is_watertight)
        df = df.append({"id":id,"class_id":class_id,"amnt_vertices":amnt_vertices,"amnt_faces":amnt_faces,"is_triangle":is_triangle,"bounding_box":bounding_box,
                        "water_tight":water_tight},ignore_index=True)
        if i == 100:  pass          
            # break
    return df

def load_dataset(load=LOAD_PATH):
    print(f"opened dataset from {load}")
    df = pd.read_csv(load)
    return df

def main():
    mesh_files = dir_to_sorted_file_list()
    df = create_data(mesh_files)
    df.to_csv(SAVE_PATH,index=False)
    print(f"Saved dataset to: {SAVE_PATH}")

if __name__=="__main__":
    main()
   