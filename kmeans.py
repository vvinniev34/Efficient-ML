import pickle
from sklearn.cluster import KMeans

with open('all_activations.pkl', 'rb') as f:
    data = pickle.load(f)

def kmeans_cluster(nclusters):
    print(f"{nclusters} clusters, converging...")
    kmeans = KMeans(n_clusters=nclusters)
    kmeans.fit(data)

    with open(f'kmeans_{nclusters}.pkl', 'wb') as f:
        pickle.dump(kmeans.cluster_centers_, f)

clusters = [128, 256, 512, 1024, 2048, 4096, 6144, 8192, 9216]
for cluster in clusters:
    kmeans_cluster(cluster)
