import pickle
import numpy as np
import matplotlib.pyplot as plt

a_file = open('data.pkl', 'rb')
data = pickle.load(a_file)

points = {}
for key in data.keys():
    if key not in points:
        points[key] = []
        frr = []
        far = []

        for threshold in np.linspace(-4000, 0, 5000):
            FRR = np.sum((data[key][0] > threshold) == 0) / len(data[key][0]))
            frr.extend([FRR])
            
            FAR = np.sum((data[key][1] < threshold) == 0) / len(data[key][1]))
            far.extend([FAR])

        points[key].extend([frr, far])


def plot_fig(points, keyword):
    FAR = points[keyword][1]
    FRR = points[keyword][0]
    plt.figure()
    plt.rcParams.update({'font.family':'monospace'})
    plt.rcParams['figure.figsize'] = 6, 4
    plt.rcParams['font.size'] = 10
    plt.plot(FAR, FRR, label=keyword, color='r')
    plt.xlabel('False Alarm Rate (%)')
    plt.ylabel('False Reject Rate (%)')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(f'../figure_evaluate/{keyword}.png')

for key in data.keys():
    plot_fig(points=points, keyword=key)