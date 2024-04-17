import pickle
import numpy as np
import queue
from sklearn.cluster import AgglomerativeClustering

# Leaves: 10000


# with open('all_activations.pkl', 'rb') as f:
#     data = np.array(pickle.load(f))
# data = np.corrcoef(data, rowvar=False)



# with open(f'test.txt', 'w', encoding='utf-8') as f:
#         f.write("")
# for line in data:
#     with open(f'test.txt', 'a', encoding='utf-8') as f:
#         f.write(f"{np.array2string(line)}\n")



# data = np.nan_to_num(data)
# print("rows: ", len(data))
# print("cols: ", len(data[0]))
# clustering = AgglomerativeClustering(linkage='ward')
# clustering.fit(data)

# print(f"Labels: {clustering.labels_}")
# print(f"Leaves: {clustering.n_leaves_}")
# print(f"Clusters: {clustering.n_clusters_}")

# children = clustering.children_
# with open(f'agglo_clustering/agglo_children.pkl', 'wb') as f:
#     pickle.dump(children, f)

with open('agglo_clustering/agglo_children.pkl', 'rb') as f:
    children = pickle.load(f)

adjList = {}
num_samples = 1024
for i in range(len(children)):
    node = children[i]
    adjList[node[0]] = i + num_samples
    adjList[node[1]] = i + num_samples

topDownAdjList = {}
for i in range(len(children)):
    node = children[i]
    topDownAdjList[num_samples + i] = node

def find_root(leaves, parent_map):
    current_node = leaves
    while current_node in parent_map:
        parent_node = parent_map[current_node]
        current_node = parent_node

    return current_node

for i in range(1024):
    root = find_root(i, adjList)
    if root != 2046:
        print(root)
root = 2046

numerators = [1,1,3,1,5,3,7]
denominators = [8,4,8,2,8,4,8]
benchmarks = []
k = 0
for i in range(len(numerators)):
    benchmarks.append(int(num_samples * (numerators[i] / denominators[i])))
print(benchmarks)

def postorder_traversal(tree, current_node):
    global k
    if current_node not in tree: 
        return [current_node]

    leaves = []
    for child in tree[current_node]:
        leaves.extend(postorder_traversal(tree, child))

    if k < len(benchmarks) and len(leaves) > benchmarks[k]:
        print(f"benchmark: {benchmarks[k]}; achieved: {len(leaves)}")
        with open(f'agglo_clustering/agglo_{numerators[k]}_{denominators[k]}.pkl', 'wb') as f:
            pickle.dump(leaves, f)
        with open(f'agglo_clustering/agglo_{numerators[k]}_{denominators[k]}.txt', 'w', encoding='utf-8') as f:
            for item in leaves:
                f.write(str(item) + "\n")
        k += 1
    # if len(leaves) >= 896:
    #     print(len(leaves))
    return leaves

leaves_of_subtree = postorder_traversal(topDownAdjList, root)


# 1/8: 128; achieved: 129 
# 1/4: 256; achieved: 257 
# 3/8: 384; achieved: 385 
# 1/2: 512; achieved: 545 
# 5/8: 640; achieved: 759 
# 3/4: 768; achieved: 1024 