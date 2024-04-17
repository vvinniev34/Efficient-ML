import pickle
import numpy as np
from sklearn.cluster import KMeans

with open('all_activations.pkl', 'rb') as f:
    data = np.array(pickle.load(f))

def extractCols(matrix, columns_to_extract):
    new_matrix = []
    for row in matrix:
        new_row = [row[i] for i in columns_to_extract]
        new_matrix.append(new_row)
    return new_matrix

def kmeans_cluster(nclusters, numerator, denominator, columns):
    print(f"{nclusters} with {numerator}/{denominator} clusters, converging...")
    
    # split_data = data[:, :(numerator * (data.shape[1] // denominator))]
    split_data = extractCols(data, columns)
    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(split_data)

    with open(f'clusters/kmeans_agglo_{nclusters}_{numerator}div{denominator}.pkl', 'wb') as f:
        pickle.dump(kmeans.cluster_centers_, f)

clusters = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 9216]
numerators = [1,1,1,3,5,3]#,7]
denominators = [2,8,4,8,8,4]#,8]

for i in range(len(clusters)):
    for j in range(len(numerators)):
        with open(f'agglo_clustering/agglo_{numerators[j]}_{denominators[j]}.pkl', 'rb') as f:
            columns = np.array(pickle.load(f))
        kmeans_cluster(clusters[i], numerators[j], denominators[j], columns)
