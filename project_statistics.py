from matplotlib.pyplot import draw
import numpy as np
import matplotlib.pyplot as plt
from trimesh.exchange.load import load
from create_dataset import LOAD_PATH, load_dataset

def stats_to_fig(df,column_stat,param):  
    '''
    calculates statistics for dataframe
    params: amnt_vertices, amnt_faces
    '''      
    data = df[param]
    sorted_vertices = np.sort(data)
    bins = np.arange(np.max(data))[::2000]  #steps on x axis
    fig,ax = plt.subplots(figsize=(15,7))
    ax.hist(sorted_vertices,bins=bins)
    ax.set_xticks(bins)

    # set title and labels
    ax.set_title(column_stat)
    ax.set_xlabel(f"number of {column_stat.split('_')[2]}")
    ax.set_ylabel("count meshes")

    # save figure
    plt.savefig(f"stats/{column_stat}")

def get_outliers(df,column_stat="amnt_vertices"):
    '''
    returns: 
    [0] index of lower_bound < 5%
    [1] index of upperbound  > 95% 
    '''
    lowerbound_05 = df[column_stat].describe([.05,.5,.95])[4]
    upperbound_95 = df[column_stat].describe([.05,.5,.95])[6]  

    # get index of low and upperbound    
    lowerbound_05_idx = df[df[column_stat]<lowerbound_05].index
    upperbound_95_idx = df[df[column_stat]>upperbound_95].index
    print(df[column_stat].describe())
    return lowerbound_05_idx[0],upperbound_95_idx[0]

def main():
    df = load_dataset()
    # raw database
    # stats_to_fig(df,"raw_amnt_vertices","amnt_vertices")
    stats_to_fig(df,"raw_amnt_faces","amnt_faces")

    df = load_dataset(load=LOAD_PATH)
    # processed database
    '''save figures for process database'''
    # stats_to_fig(df,"normalized_amnt_vertices","amnt_vertices")
    # stats_to_fig(df,"normalized_amnt_faces","amnt_faces")

LOAD_PATH="database/normalized_db.csv"
if __name__ == "__main__":
    main()

    df = load_dataset(load = LOAD_PATH)
    print(df[df["amnt_vertices"]>2000])
    # print(get_outliers(df,"amnt_faces"))

