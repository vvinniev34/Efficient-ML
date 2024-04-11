import matplotlib.pyplot as plt
import numpy as np

full = {
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

half = {
    10000: 0.6813686490058899,
    128: 0.5694224834442139,
    256: 0.589992105960846,
    512: 0.5985957384109497,
    1024: 0.615407407283783,
    2048: 0.6331091523170471,
    4096: 0.6535798907279968,
    6144: 0.6664358973503113,
    8192: 0.6794897317886353,
    9216: 0.6804786324501038,
}

fourth = {
    128: 0.6549643874168396,
    256:  0.654469907283783,
    512: 0.6607001423835754,
    1024: 0.6629746556282043,
    2048: 0.6695016026496887,
    4096: 0.6732594966888428,
    6144:  0.677907407283783,
    8192:  0.681467592716217,
    9216: 0.6823576092720032,
    10000: 0.6813686490058899,
}


eighth = {
    10000: 0.6813686490058899,
    128: 0.6709849834442139,
    256: 0.6731606125831604,
    512: 0.6738528609275818,
    1024: 0.675039529800415,
    2048:0.6744462251663208,
    4096: 0.6784018874168396,
    6144: 0.6782041192054749,
    8192: 0.6809731125831604,
    9216: 0.6807753443717957,
}

three_eighth = {
    10000: 0.6813686490058899,
    128: 0.6182753443717957,
    256: 0.6235166192054749,
    512: 0.6342958807945251,
    1024: 0.6439873576164246,
    2048: 0.651602029800415,
    4096: 0.6665347814559937,
    6144: 0.6742483973503113,
    8192: 0.6799841523170471,
    9216: 0.6803797483444214,
}

five_eighth = {
    10000: 0.6813686490058899,
    128: 0.5281843543052673,
    256: 0.5488528609275818,
    512: 0.5682357549667358,
    1024: 0.5855419039726257,
    2048: 0.6118472814559937,
    4096: 0.6401305198669434,
    6144:0.6605023741722107,
    8192: 0.6755340099334717,
    9216: 0.6817642450332642,
}

three_fourth = {
    10000: 0.6813686490058899,
    128: 0.49367088079452515,
    256: 0.5341178774833679,
    512: 0.5518196225166321,
    1024: 0.5669501423835754,
    2048: 0.5940466523170471,
    4096: 0.6286590099334717,
    6144: 0.6571400165557861,
    8192: 0.6751384735107422,
    9216: 0.6816653609275818
}

seven_eighth = {
    10000: 0.6813686490058899,
    128: 0.45589399337768555,
    256: 0.49782437086105347,
    512: 0.5184928774833679,
    1024: 0.5477650165557861,
    2048: 0.5746637582778931,
    4096: 0.613132894039154,
    6144: 0.6475474834442139,
    8192: 0.671875,
    9216: 0.6785996556282043
}


clusters = np.array([128, 256, 512, 1024, 2048, 4096, 6144, 8192, 9216, 10000])
accuracies_full = []
accuracies_half = []
accuracies_fourth = []
accuracies_eighth = []
accuracies_three_eighth = []
accuracies_five_eighth = []
accuracies_three_fourth = []
accuracies_seven_eighth = []
accuracies = [accuracies_full, accuracies_half, accuracies_fourth, accuracies_eighth, accuracies_three_eighth, accuracies_five_eighth, accuracies_three_fourth, accuracies_seven_eighth]
for cluster in clusters:
    accuracies_full.append(full[cluster] * 100)
    accuracies_half.append(half[cluster] * 100)
    accuracies_fourth.append(fourth[cluster] * 100)
    accuracies_eighth.append(eighth[cluster] * 100)
    accuracies_three_eighth.append(three_eighth[cluster] * 100)
    accuracies_five_eighth.append(five_eighth[cluster] * 100)
    accuracies_three_fourth.append(three_fourth[cluster] * 100)
    accuracies_seven_eighth.append(seven_eighth[cluster] * 100)
for accuracy in accuracies:
    accuracy = np.array(accuracy)


differences_full = []
differences_half = []
differences_fourth = []
differences_eighth = []
differences_three_eighth = []
differences_five_eighth = []
differences_three_fourth = []
differences_seven_eighth = []
differences = [differences_full, differences_half, differences_fourth, differences_eighth, differences_three_eighth, differences_five_eighth, differences_three_fourth, differences_seven_eighth]
for cluster in clusters:
    differences_full.append((full[10000] * 100) - (full[cluster] * 100))
    differences_half.append((half[10000] * 100) - (half[cluster] * 100))
    differences_fourth.append((fourth[10000] * 100) - (fourth[cluster] * 100))
    differences_eighth.append((eighth[10000] * 100) - (eighth[cluster] * 100))
    differences_three_eighth.append((three_eighth[10000] * 100) - (three_eighth[cluster] * 100))
    differences_five_eighth.append((five_eighth[10000] * 100) - (five_eighth[cluster] * 100))
    differences_three_fourth.append((three_fourth[10000] * 100) - (three_fourth[cluster] * 100))
    differences_seven_eighth.append((seven_eighth[10000] * 100) - (seven_eighth[cluster] * 100))
for index, difference in enumerate(differences):
    differences[index] = np.array(difference)

# p = np.polyfit(np.log(clusters), accuracies, 1)
# m, b = p[0], p[1]


# plt.scatter(clusters, accuracies, label='Data')
# plt.plot(clusters, m*np.log(clusters) + b, color='red', label='Trendline')
# plt.xlabel('k Clusters')
# plt.ylabel('Accuracies (%)')
# plt.title('Accuracies vs k Clusters')
# plt.grid(True)
# plt.legend()
# plt.show()

p_full = np.polyfit(np.log(clusters), differences_full, 1)
m_full, b_full = p_full[0], p_full[1]

p_half = np.polyfit(np.log(clusters), differences_half, 1)
m_half, b_half = p_half[0], p_half[1]

p_fourth = np.polyfit(np.log(clusters), differences_fourth, 1)
m_fourth, b_fourth = p_fourth[0], p_fourth[1]

p_eighth = np.polyfit(np.log(clusters), differences_eighth, 1)
m_eighth, b_eighth = p_eighth[0], p_eighth[1]

p_three_eighth = np.polyfit(np.log(clusters), differences_three_eighth, 1)
m_three_eighth, b_three_eighth = p_three_eighth[0], p_three_eighth[1]

p_five_eighth = np.polyfit(np.log(clusters), differences_five_eighth, 1)
m_five_eighth, b_five_eighth = p_five_eighth[0], p_five_eighth[1]

p_three_fourth = np.polyfit(np.log(clusters), differences_three_fourth, 1)
m_three_fourth, b_three_fourth = p_three_fourth[0], p_three_fourth[1]

p_seven_eighth = np.polyfit(np.log(clusters), differences_seven_eighth, 1)
m_seven_eighth, b_seven_eighth = p_seven_eighth[0], p_seven_eighth[1]

plt.scatter(clusters, differences_full, label='full', color='red')
plt.plot(clusters, m_full*np.log(clusters) + b_full, color='red', label='full Trendline')

plt.scatter(clusters, differences_half, label='1/2', color='blue')
plt.plot(clusters, m_half*np.log(clusters) + b_half, color='blue', label='1/2 Trendline')

plt.scatter(clusters, differences_fourth, label='1/4', color='green')
plt.plot(clusters, m_fourth*np.log(clusters) + b_fourth, color='green', label='1/4 Trendline')

plt.scatter(clusters, differences_eighth, label='1/8', color='orange')
plt.plot(clusters, m_eighth*np.log(clusters) + b_eighth, color='orange', label='1/8 Trendline')

plt.scatter(clusters, differences_three_eighth, label='3/8', color='purple')
plt.plot(clusters, m_three_eighth*np.log(clusters) + b_three_eighth, color='purple', label='3/8 Trendline')

plt.scatter(clusters, differences_five_eighth, label='5/8', color='cyan')
plt.plot(clusters, m_five_eighth*np.log(clusters) + b_five_eighth, color='cyan', label='5/8 Trendline')

plt.scatter(clusters, differences_three_fourth, label='3/4', color='yellow')
plt.plot(clusters, m_three_fourth*np.log(clusters) + b_three_fourth, color='yellow', label='3/4 Trendline')

plt.scatter(clusters, differences_seven_eighth, label='7/8', color='olive')
plt.plot(clusters, m_seven_eighth*np.log(clusters) + b_seven_eighth, color='olive', label='7/8 Trendline')

plt.xlabel('k Clusters')
plt.ylabel('Dif Accuracies To Original (%)')
plt.title('Dif Accuracies To Original vs k Clusters')
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('dif_accuracy_over_clusters')
