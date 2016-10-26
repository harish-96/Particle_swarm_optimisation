import numpy as np
import struct
import matplotlib.pyplot as plt
import random
import scipy.io as sio
import tarfile
import os
from pso import *
import pdb


def unpack_dat(imgpath, labpath):
    """ Unpack images and labels obtained online from
    http://yann.lecun.com/exdb/mnist/
    """
    with open(labpath, 'rb') as f:
        magic_no, n_dim = struct.unpack('>ii', f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)

    with open(imgpath, 'rb') as f:
        magic_num, n_dim, n_rows, n_cols = struct.unpack(">iiii",
                                                         f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(len(labels), 784)

    images = [np.reshape(x, (784, 1)) for x in images]
    labels = [np.array([y == i for i in range(10)])[np.newaxis].T
              for y in labels]

    return images, labels


def display_data(imgs, nrows=1, ncols=1, nx_pixels=28, ny_pixels=28):
    """Dispay the images given in X. 'nrows' and 'ncols' are
    the number of rows and columns in the displayed data"""
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,)

    if (nrows + ncols) == 2:
        ax.imshow(imgs[0].reshape(nx_pixels, ny_pixels),
                  cmap='Greys', interpolation="bicubic")
    else:
        ax = ax.flatten()
        for i in range(nrows * ncols):
            ax[i].imshow(imgs[i].reshape(nx_pixels, ny_pixels),
                         cmap='Greys', interpolation="bicubic")
    plt.tight_layout()
    plt.show()


def sigmoid(z):
    """Evaluates the sigmoid function at the given input
    Returns a numpy array"""

    z = np.asarray(z)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    """Evaluates the derivative of the sigmoid function at the given input
    Returns a numpy array"""

    return sigmoid(z) * (1 - sigmoid(z))


class NN_hwr(object):
    """A template class for a neural network to recognise handwritten text
    Initialise with a list with each element being the number of neurons in
    that layer
    For example: neural_net = NN_hwr([400, 30, 10]) creates a neural network
    with 3 layers and 400, 30 and 10 as their sizes"""

    def __init__(self, num_neurons_list):
        """Input must be a list of numbers."""

        for i in num_neurons_list:
            if type(i) not in [type(2)]:
                raise TypeError("Expected integer type")
        
        self.num_layers = len(num_neurons_list)
        self.num_neurons_list = num_neurons_list
        self.biases = [np.random.randn(y, 1) for y in num_neurons_list[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(num_neurons_list[:-1],
                        num_neurons_list[1:])]

    def forward_prop(self, x_train):
        """Computes the activations and weighted inputs of the neurons in
        the network for the given input data
        Returns a tuple of lists containing activations and weighted inputs"""

        activations = []
        z = []
        activations.append(x_train)

        for i in range(self.num_layers - 1):
            z.append(np.dot(self.weights[i], activations[-1]) + self.biases[i])
            activations.append(sigmoid(z[-1]))

        return activations[1:], z

    def train_batch_pso(self, batch, exit_error=500):
        error_function = lambda positions: self.swarm_cost_function(positions, batch)
        training_swarm = Swarm(100, error_function, exit_error=exit_error)
        training_swarm.optimise()
        position = training_swarm.gbest
        self.weights[0] = np.reshape(position[0], (784, 15))
        self.weights[1] = np.reshape(position[1], (15, 10))
        self.biases[0] = np.reshape(position[2], (15, 1))
        self.biases[0] = np.reshape(position[3], (10, 1))
        

    def train_nn(self, X_train, y_train, n_epochs, batch_size, learning_rate):
        """Trains the neural network with the test data. n_epochs is the number
        sweeps over the whole data. batch_size is the number of training example per
        batch in the stochastic gradient descent. X_train and y_train are the images and
        labels in the training data. X_train must be a 2-D array with only one row and 
        y_train is an array of length 10 of zeros everywhere except at the image 
        label (where there is a 1)"""

        m = len(y_train)
        train_data = list(zip(X_train, y_train))

        for i in range(n_epochs):
            random.shuffle(train_data)
            batches = [train_data[j:j + batch_size]
                       for j in range(0, m, batch_size)]
            for batch in batches:
#                self.train_batch(batch, learning_rate)
                self.train_batch_pso(batch)
            print("epoch no: %d" % i, self.cost_function(X_train, y_train))

    def cost_function(self, X_train, y_train):
        J = 0
        for i in range(len(y_train)):
            J += 0.5 * np.sum((self.forward_prop(X_train[i])[0][-1] - y_train[i])**2)
        return J
        
    def swarm_cost_function(self, position, batch):
        x, y = zip(*batch)
        pdb.set_trace()
        self.weights[0] = np.reshape(position[0], (784, 15))
        self.weights[1] = np.reshape(position[1], (15, 10))
        self.biases[0] = np.reshape(position[2], (15, 1))
        self.biases[0] = np.reshape(position[3], (10, 1))
        return self.cost_function(x, y)


def load_data(path):
    """Loads the image data from the path provided and returns the images and labels"""
    if os.path.splitext(path)[1] == '.gz':
        tfile = tarfile.open(path)
        tfile.extractall("./data/")
        tfile.close()
        path = os.path.splitext(os.path.splitext(path)[0])[0]
    data_dict = sio.loadmat(path)
    if 'train' in path:
        return data_dict['X_train'], data_dict['y_train']
    else:
        return data_dict['X_test'], data_dict['y_test']


if __name__ == '__main__':
    X_train, y_train = load_data("./data/traindata.mat.tar.gz")
    X_train = X_train[:500]
    y_train = y_train[:500]
    X_test, y_test = load_data("./data/testdata.mat.tar.gz")
    display_data(X_train[:10], 2, 5)

    nn = NN_hwr([len(X_train[0]), 15, 10])
    nn.train_nn(X_train, y_train, 10, 20, 0.06)

    accuracy = 0
    for i in range(len(X_test[:100])):
        out = nn.forward_prop(X_test[i])[0][-1]
        if np.argmax(out) == np.where(y_test[i])[0][0]:
            accuracy += 1
    print("accuracy: ", accuracy)
