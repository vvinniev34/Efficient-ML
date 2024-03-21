import pickle
from sklearn.cluster import KMeans

with open('all_activations.pkl', 'rb') as f:
    data = pickle.load(f)

print("128 clusters, converging...")
kmeans_128 = KMeans(n_clusters=128)
kmeans_128.fit(data)
print("256 clusters, converging...")
kmeans_256 = KMeans(n_clusters=256)
kmeans_256.fit(data)
print("512 clusters, converging...")
kmeans_512 = KMeans(n_clusters=512)
kmeans_512.fit(data)

with open('kmeans_128.pkl', 'wb') as f:
    pickle.dump(kmeans_128.cluster_centers_, f)

with open('kmeans_256.pkl', 'wb') as f:
    pickle.dump(kmeans_256.cluster_centers_, f)

with open('kmeans_512.pkl', 'wb') as f:
    pickle.dump(kmeans_512.cluster_centers_, f)
