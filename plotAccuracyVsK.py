import matplotlib.pyplot as plt
import numpy as np

clustersToAcc = {
    128: 0.42662182450294495,
    256: 0.45559731125831604,
    512: 0.4879351258277893,
    1024: 0.5116692781448364,
    2048: 0.548655092716217,
    4096: 0.6017602682113647,
    6144: 0.6420094966888428,
    8192: 0.6683148741722107,
    9216: 0.6782041192054749,
    10000: 0.6813686490058899
}

clusters = np.array([128, 256, 512, 1024, 2048, 4096, 6144, 8192, 9216, 10000])
accuracies = []
for cluster in clusters:
    accuracies.append(clustersToAcc[cluster] * 100)
accuracies = np.array(accuracies)
differences = []
for cluster in clusters:
    differences.append((clustersToAcc[10000] * 100) - (clustersToAcc[cluster] * 100))
differences = np.array(differences)

p = np.polyfit(np.log(clusters), accuracies, 1)
m, b = p[0], p[1]

plt.scatter(clusters, accuracies, label='Data')
plt.plot(clusters, m*np.log(clusters) + b, color='red', label='Trendline')
plt.xlabel('k Clusters')
plt.ylabel('Accuracies (%)')
plt.title('Accuracies vs k Clusters')
plt.grid(True)
plt.legend()
plt.show()

# p = np.polyfit(np.log(clusters), differences, 1)
# m, b = p[0], p[1]

# plt.scatter(clusters, differences, label='Data')
# plt.plot(clusters, m*np.log(clusters) + b, color='red', label='Trendline')
# plt.xlabel('k Clusters')
# plt.ylabel('Dif Accuracies To Original (%)')
# plt.title('Dif Accuracies To Original vs k Clusters')
# plt.grid(True)
# plt.legend()
# plt.show()
