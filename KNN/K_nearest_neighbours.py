#Three lines to make our compiler able to draw:
import sys
import matplotlib
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

means = [[2, 4, 9], [6, 5, 6], [9, 9, 9]]
cov = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
N = 20 # Number of elements in each class
K = 3 # Number of classes
Y = [[] for _ in range(K)]
Y[0] = np.random.multivariate_normal(means[0], cov, N)
Y[1] = np.random.multivariate_normal(means[1], cov, N)
Y[2] = np.random.multivariate_normal(means[2], cov, N)
# Creating a dataset by concatenating 3 vectors Y1, Y2, Y3 based on the given average(mean) point and the 3-dimensional base(coverance)

X = np.concatenate((Y[0], Y[1], Y[2]), axis = 0)
classes = []

for num in range(K):
    for _ in range(len(Y[num])):
        classes.append(num)

knn = KNeighborsClassifier(n_neighbors=10, weights = 'distance')
knn.fit(X, classes)
# Creating the model using KNeighborsClassifier algorithm

new_point = np.array([[6, 3, 6]])
color_dict = {0 : 'purple', 1 : 'red', 2 : 'green', 3 : 'black'}
classes.append(3)
# new point have the black color, different from other points from other classes

prediction = knn.predict(new_point) # Predicting the black point to determine which class it belongs to :3

# Illustrating the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(np.concatenate((X[:, 0], new_point[:, 0])),
           np.concatenate((X[:, 1], new_point[:, 1])),
           np.concatenate((X[:, 2], new_point[:, 2])),
           c=[color_dict[it] for it in classes])

ax.text(x=new_point[0][0] - 0.7, y=new_point[0][1] - 0.7, z=new_point[0][2] - 0.7, s=f"new point, class: {color_dict[prediction[0]]}")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('K Nearest Neighbours')
ax.legend()
plt.show()

