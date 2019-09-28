import numpy as np
import h5py
import copy


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
        W = [None, ]
        b = [None, ]
        neuron_dim_list = (n_x,) + self.neuron_dims

        if self.mid_activation == "relu":
            adj = 2
        else:
            adj = 1

        for l in range(1, L+1):
            W.append(np.random.randn(neuron_dim_list[l-1], neuron_dim_list[l]) * np.sqrt(adj/neuron_dim_list[l-1]))
            #self.W.append(np.random.randn(neuron_dim_list[l], neuron_dim_list[l-1]).T / np.sqrt(neuron_dim_list[l-1])) #this one will produce the same results as Coursera
            b.append(np.zeros((1, neuron_dim_list[l])))

        return W, b

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

    def forward_prop(self, data: DataSet, W, b):
        L = len(self.neuron_dims)
        X, Y = data.get_data()
        Z = [None, ]
        A = [X, ]
        D = None

        if 1 > self.keep_prob > 0:
            D = [np.ones(X.shape), ]

        for l in range(1, L+1):
            Z.append(np.dot(A[l-1], W[l]) + b[l])
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

        return A, D

    def calculate_cost(self, data: DataSet, A, W):
        L = len(self.neuron_dims)
        X, Y = data.get_data()
        m = data.get_m()
        J = (-1 / m) * (np.dot(Y.T, np.log(A[L])) + np.dot((1 - Y).T, np.log(1 - A[L]))).squeeze()

        if self.lam != 0:
            for l in range(1, L + 1):
                J += (self.lam/(2*m))*np.linalg.norm(W[l])**2

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

    def test_grads(self, data: DataSet, W, b, dW, db, epsilon):
        L = len(self.neuron_dims)

        if L > 1:
            min_length = L-1
        else:
            min_length = L

        dW_test = [None] * (L+1)
        for l in range(min_length, L + 1):
            dW_test[l] = np.zeros(W[l].shape)
            for r in range(W[l].shape[0]):
                for c in range(W[l].shape[1]):
                    W_plus_test = copy.deepcopy(W)
                    W_plus_test[l][r,c] += epsilon
                    A, D = self.forward_prop(data, W_plus_test, b)
                    J_plus = self.calculate_cost(data, A, W_plus_test)

                    W_minus_test = copy.deepcopy(W)
                    W_minus_test[l][r, c] -= epsilon
                    A, D = self.forward_prop(data, W_minus_test, b)
                    J_minus = self.calculate_cost(data, A, W_minus_test)

                    dW_test[l][r, c] = (J_plus - J_minus) / (2*epsilon)
            dW_test[l] = dW_test[l].T
            dW_delta = np.linalg.norm(dW[l] - dW_test[l]) / (np.linalg.norm(dW[l]) + np.linalg.norm(dW_test[l]))
            if dW_delta > epsilon:
                print("GRAD FAIL: dW_delta=" + str(dW_delta))
            else:
                print("GRAD PASS: dW_delta=" + str(dW_delta))

        db_test = [None] * (L+1)
        for l in range(min_length, L + 1):
            db_test[l] = np.zeros(b[l].shape)
            for r in range(b[l].shape[0]):
                for c in range(b[l].shape[1]):
                    b_plus_test = copy.deepcopy(b)
                    b_plus_test[l][r,c] += epsilon
                    A, D = self.forward_prop(data, W, b_plus_test)
                    J_plus = self.calculate_cost(data, A, W)

                    b_minus_test = copy.deepcopy(b)
                    b_minus_test[l][r, c] -= epsilon
                    A, D = self.forward_prop(data, W, b_minus_test)
                    J_minus = self.calculate_cost(data, A, W)

                    db_test[l][r, c] = (J_plus - J_minus) / (2*epsilon)
            db_test[l] = db_test[l].T
            db_delta = np.linalg.norm(db[l] - db_test[l]) / (np.linalg.norm(db[l]) + np.linalg.norm(db_test[l]))
            if db_delta > epsilon:
                print("GRAD FAIL: db_delta=" + str(db_delta))
            else:
                print("GRAD PASS: db_delta=" + str(db_delta))


    def backward_prop(self, data: DataSet, A, D, W, b):
        L = len(self.neuron_dims)
        m = data.get_m()
        X, Y = data.get_data()
        dA = (-1/m)*np.divide((Y-A[L]), A[L]*(1-A[L])).T
        dg = NeuralNet.d_g(self.final_activation, A[L].T)
        dZ = [dA*dg, ]
        db = [np.sum(dZ[0], axis=1, keepdims=True), ]
        dW = [np.dot(dZ[0], A[L-1]), ]

        if self.lam != 0:
            dW[0] += (self.lam/m)*W[L].T

        for l in reversed(range(1, L)):
            dg = NeuralNet.d_g(self.mid_activation, A[l].T)
            dZ.insert(0, np.dot(W[l+1], dZ[0])*dg)

            if 0 < self.keep_prob < 1:
                dZ[0] *= D[l].T/self.keep_prob

            db.insert(0, np.sum(dZ[0], axis=1, keepdims=True))
            dW.insert(0, np.dot(dZ[0], A[l-1]))

            if self.lam != 0:
                dW[0] += (self.lam / m) * W[l].T

        dW.insert(0, None)
        db.insert(0, None)
        return dW, db

    def update_parameters(self, rate, dW, db):
        L = len(self.neuron_dims)
        for l in range(1, L+1):
            self.W[l] -= dW[l].T*rate
            self.b[l] -= db[l].T*rate

    def train(self, data: DataSet, iterations: int=1000, rate: float=0.0075, epsilon: float=0):
        old_keep_prob = self.keep_prob
        self.keep_prob = 1
        n = data.get_n()
        self.W, self.b = self.initialize_parameters(n)

        for i in range(iterations):
            A, D = self.forward_prop(data, self.W, self.b)
            J = self.calculate_cost(data, A, self.W)

            dW, db = self.backward_prop(data, A, D, self.W, self.b)

            if epsilon != 0:
                self.test_grads(data, self.W, self.b, dW, db, epsilon)

            print(str(i) + ": " + str(J))
            self.update_parameters(rate, dW, db)

        self.keep_prob = old_keep_prob

    def test(self, data: DataSet):
        X, Y = data.get_data()
        A, D = self.forward_prop(data, self.W, self.b)
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
nn.train(train_data, iterations=2500, rate=0.0075) #, epsilon=1e-7

print(nn.test(train_data))
print(nn.test(test_data))