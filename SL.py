import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

labels = kmeans.predict(X)

plt.figure(figsize=(8, 6))

colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, [0, 1, 2], kmeans.cluster_centers_):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], color=color, alpha=.8, lw=2,
                label=f'Cluster {i+1}, Center: {target_name}')

plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('KMeans Clustering on Synthetic Blobs Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
