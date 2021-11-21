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
    
    df = df[df["amnt_vertices"]<5500]
    data = df[param]
    sorted_vertices = np.sort(data)
    bins = np.arange(np.max(data))[::500]  #steps on x axis
    print(np.max(data))

    fig,ax = plt.subplots(figsize=(15,7))
    ax.hist(sorted_vertices,bins=bins)
    ax.set_xticks(bins)
    # ax.set
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    

    # set title and labels
    ax.set_title(column_stat,fontsize=20)
    ax.set_xlabel(f"number of {column_stat.split('_')[2]}",fontsize=20)
    ax.set_ylabel("count meshes",fontsize=20)

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
    stats_to_fig(df,"normalized_amnt_vertices","amnt_vertices")
    # stats_to_fig(df,"raw_amnt_faces","amnt_faces")

    df = load_dataset(load=LOAD_PATH)
    # processed database
    '''save figures for process database'''
    # stats_to_fig(df,"normalized_amnt_vertices","amnt_vertices")
    # stats_to_fig(df,"normalized_amnt_faces","amnt_faces")

# LOAD_PATH="database/normalized_db.csv"
if __name__ == "__main__":
    main()
    # df = load_dataset()
    # print(df["amnt_faces"].max())
    # print(df[df["amnt_vertices"]>75000])

    # df = load_dataset(load = LOAD_PATH)
    # print(df[(df["amnt_vertices"]<577) ])

    # print(df[df["amnt_vertices"]>3882])
    # print(get_outliers(df,"amnt_faces"))

