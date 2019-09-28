import numpy as np
import h5py


class DataSet:

    def __init__(self, X: np.ndarray, Y: np.ndarray = None):
        self.X = X
        self.Y = Y

    def set_data(self, X: np.ndarray, Y: np.ndarray = None):
        self.X = X
        self.Y = Y

    def get_data(self):
        return self.X, self.Y

    def get_m(self):
        return self.X.shape[0]

    def get_n(self):
        return self.X.shape[1]


class NeuralNet:

    def __init__(self, neurons_dims: tuple, mid_activation: str = "relu", final_activation: str = "sigmoid", lam: float=0, keep_prob: float=1):
        self.neuron_dims = neurons_dims
        self.mid_activation = mid_activation
        self.final_activation = final_activation
        self.lam = lam
        self.keep_prob = keep_prob

    def initialize_parameters(self, n_x: int):
        np.random.seed(1)
        L = len(self.neuron_dims)
        self.W = [None, ]
        self.b = [None, ]
        neuron_dim_list = (n_x,) + self.neuron_dims

        if self.mid_activation == "relu":
            adj = 2
        else:
            adj = 1

        for l in range(1, L+1):
            self.W.append(np.random.randn(neuron_dim_list[l-1], neuron_dim_list[l]) * np.sqrt(adj/neuron_dim_list[l-1]))
            #self.W.append(np.random.randn(neuron_dim_list[l], neuron_dim_list[l-1]).T / np.sqrt(neuron_dim_list[l-1])) #this one will produce the same results as Coursera
            self.b.append(np.zeros((1, neuron_dim_list[l])))

    @staticmethod
    def sigmoid(z: np.ndarray):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def relu(z: np.ndarray):
        return np.maximum(0, z)

    @staticmethod
    def activate(activation: str, Z: np.ndarray):
        if activation == "sigmoid":
            A = NeuralNet.sigmoid(Z)
        elif activation == "relu":
            A = NeuralNet.relu(Z)
        elif activation == "tanh":
            A = np.tanh(Z)
        # Should add an exception if the activation isn't an anticipated value
        return A

    def forward_prop(self, data: DataSet):
        L = len(self.neuron_dims)
        X, Y = data.get_data()
        Z = [None, ]
        A = [X, ]
        D = None

        if 1 > self.keep_prob > 0:
            D = [np.ones(X.shape), ]

        for l in range(1, L+1):
            Z.append(np.dot(A[l-1], self.W[l]) + self.b[l])
            if l == L:
                A.append(NeuralNet.activate(self.final_activation, Z[l]))

                if 1 > self.keep_prob > 0:
                    D.append(np.ones(A[l].shape))
            else:
                A.append(NeuralNet.activate(self.mid_activation, Z[l]))

                if 1 > self.keep_prob > 0:
                    D.append(np.random.rand(A[l].shape[0], A[l].shape[1]))
                    D[l] = D[l] < self.keep_prob
                    A[l] *= D[l]/self.keep_prob

        return Z, A, D

    def calculate_cost(self, data: DataSet):
        L = len(self.neuron_dims)
        X, Y = data.get_data()
        m = data.get_m()
        J = (-1 / m) * (np.dot(Y.T, np.log(self.A[L])) + np.dot((1 - Y).T, np.log(1 - self.A[L]))).squeeze()

        if self.lam != 0:
            for l in range(1, L + 1):
                J += (self.lam/(2*m))*np.linalg.norm(self.W[l])**2

        return J

    @staticmethod
    def d_sigmoid(A: np.ndarray):
        return A*(1-A)

    @staticmethod
    def d_relu(A: np.ndarray):
        dg = np.zeros(A.shape)
        dg[A>0] = 1
        return dg

    @staticmethod
    def d_tanh(A: np.ndarray):
        return 1-A**2

    @staticmethod
    def d_g(activation: str, A: np.ndarray):
        if activation == "sigmoid":
            dg = NeuralNet.d_sigmoid(A)
        elif activation == "relu":
            dg = NeuralNet.d_relu(A)
        elif activation == "tanh":
            dg = NeuralNet.d_tanh(A)
        # Should add an exception if the activation isn't an anticipated value
        return dg

    def backward_prop(self, data: DataSet):
        L = len(self.neuron_dims)
        m = data.get_m()
        X, Y = data.get_data()
        dA = (-1/m)*np.divide((Y-self.A[L]), self.A[L]*(1-self.A[L])).T
        dg = NeuralNet.d_g(self.final_activation, self.A[L].T)
        dZ = [dA*dg, ]
        db = [np.sum(dZ[0], axis=1, keepdims=True), ]
        dW = [np.dot(dZ[0], self.A[L-1]), ]

        if self.lam != 0:
            dW[0] += (self.lam/m)*self.W[L].T

        for l in reversed(range(1, L)):
            dg = NeuralNet.d_g(self.mid_activation, self.A[l].T)
            dZ.insert(0, np.dot(self.W[l+1], dZ[0])*dg)

            if 0 < self.keep_prob < 1:
                dZ[0] *= self.D[l].T/self.keep_prob

            db.insert(0, np.sum(dZ[0], axis=1, keepdims=True))
            dW.insert(0, np.dot(dZ[0], self.A[l-1]))

            if self.lam != 0:
                dW[0] += (self.lam / m) * self.W[l].T

        dZ.insert(0, None)
        dW.insert(0, None)
        db.insert(0, None)
        return dZ, dW, db

    def update_parameters(self, rate):
        L = len(self.neuron_dims)
        for l in range(1, L+1):
            self.W[l] -= self.dW[l].T*rate
            self.b[l] -= self.db[l].T*rate

    def train(self, data: DataSet, iterations: int=1000, rate: float=0.0075, epsilon: float=0):
        old_keep_prob = self.keep_prob
        self.keep_prob = 1
        n = data.get_n()
        self.initialize_parameters(n)

        for i in range(iterations):
            self.Z, self.A, self.D = self.forward_prop(data)
            self.J = self.calculate_cost(data)
            print(str(i) + ": " + str(self.J))
            self.dZ, self.dW, self.db = self.backward_prop(data)
            self.update_parameters(rate)

        self.keep_prob = old_keep_prob

    def test(self, data: DataSet):
        X, Y = data.get_data()
        Z, A, D = self.forward_prop(data)
        L = len(self.neuron_dims)
        Y_hat = np.round(A[L])
        return 1 - np.sum(np.abs(Y_hat - Y), axis=0).squeeze()/Y.size

    def predict(self, X: np.ndarray):
        pass

train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)/255
train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels
train_set_y_orig = train_set_y_orig.reshape((train_set_y_orig.shape[0]), 1)

test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)/255
test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels
test_set_y_orig = test_set_y_orig.reshape((test_set_y_orig.shape[0]), 1)

train_data = DataSet(train_set_x_orig, train_set_y_orig)
test_data = DataSet(test_set_x_orig, test_set_y_orig)

nn = NeuralNet((50,25,10,1), lam=1, keep_prob=1)
nn.train(train_data, iterations=2500, rate=0.0075, epsilon=1e-7)

print(nn.test(train_data))
print(nn.test(test_data))