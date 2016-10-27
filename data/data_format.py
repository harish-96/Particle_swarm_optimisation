import scipy.io as sio
import numpy as np
X = []
y = []
with open("iris.data") as f:
    for line in f.readlines():
        fields = str.split(line, ',')
        if len(fields) != 5:
            break
        x1 = np.asarray(np.reshape(fields[:-1], (4, 1)), dtype=float)
        X.append(x1)

        if fields[-1] == 'Iris-setosa\n':
            y.append([[1], [0], [0]])
        if fields[-1] == 'Iris-versicolor\n':
            y.append([[0], [1], [0]])
        if fields[-1] == 'Iris-virginica\n':
            y.append([[0], [0], [1]])

x = X[:-1]
X_train = (x - np.mean(x)) / (np.max(x) - np.min(x))
sio.savemat('iris_2.mat', {'X_train': X_train, 'y_train': y})
