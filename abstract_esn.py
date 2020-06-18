import numpy as np
import cvxpy as cp

class AbstractESN():
    def __init__(self, n_neurons, n_inputs, n_outputs, connectivity, washout, seed, a, spectral_radius):
        self.size = n_neurons
        self.input_size = n_inputs
        self.output_size = n_outputs
        self.connectivity = connectivity
        self.washout = washout
        self.seed = seed
        self.a = a
        self.xc = np.zeros((n_neurons,))
        self.xr = np.zeros((n_neurons,))
        self.W_abstract = np.zeros((n_neurons + n_inputs + 1, n_outputs))
        self.W_classical = np.zeros((n_neurons + n_inputs + 1, n_outputs))
        self.optimization_result = None
        self.spectral_radius = spectral_radius

        np.random.seed(seed)
        W = np.random.rand(n_neurons ** 2) - 0.5
        maskW = np.random.rand(n_neurons ** 2) > connectivity
        W[maskW] = 0.0
        W = np.reshape(W, (n_neurons, n_neurons))
        rhoW = max(abs(np.linalg.eig(W)[0]))
        W *= (spectral_radius / rhoW)

        Win = np.random.rand(n_neurons, n_inputs + 1) - 0.5

        self.W = W
        self.Win = Win
        self.ctanh = np.vectorize(self.interval_tanh)

    def interval_tanh(self, c, b):
        nc = np.tanh(c)
        nb = np.tanh(c + np.absolute(b)) - nc
        return (nc, nb)

    def interval_dot(self, c, b, A):
        nc = np.dot(A, c)
        nb = np.dot(np.absolute(A), b)
        return (nc, nb)

    def step(self, uc, ur):
        (Winc, Winr) = self.interval_dot(np.hstack((1, uc)), np.hstack((0, ur)), self.Win)
        (Wxc, Wxr) = self.interval_dot(self.xc, self.xr, self.W)
        nxc = self.a * self.xc + (1 - self.a) * (Winc + Wxc)
        nxr = self.a * self.xr + (1 - self.a) * (Winr + Wxr)
        (self.xc, self.xr) = self.ctanh(nxc, nxr)
        return (self.xc, self.xr)

    def calculate_classical_weights(self, y, X):
        for i in range(self.output_size):
            if self.output_size > 1:
                self.W_classical[:, i] = np.dot(np.dot(y[:, i], X), np.linalg.inv(np.dot(X.T, X) + 1e-8 * np.eye(1 + self.input_size + self.size)))
            else:
                self.W_classical[:, i] = np.dot(np.dot(y, X),
                                                np.linalg.inv(np.dot(X.T, X) + 1e-8 * np.eye(1 + self.input_size + self.size)))
        return self.W_classical

    def calculate_abstract_weights(self, y, dy, X, XR):
        self.optimization_result = []
        for i in range(self.output_size):
            if self.output_size > 1:
                w = cp.Variable(self.size + self.input_size + 1)
                objective = cp.Minimize(cp.sum_squares(X * w - y[:, i]))
                constraints = [XR * cp.abs(w) <= dy[:, i] - np.nextafter(0, 1)]
            else:
                w = cp.Variable(self.size + self.input_size + 1)
                objective = cp.Minimize(cp.sum_squares(X * w - y))
                constraints = [XR * cp.abs(w) <= dy - np.nextafter(0, 1)]

            prob = cp.Problem(objective, constraints)

            self.optimization_result.append(prob.solve(solver=cp.SCS))
            self.W_abstract[:, i] = w.value
        return self.W_abstract

    def predict_abstract(self, u, du):
        return self.interval_dot(np.hstack((1, u, self.xc)), np.hstack((0, du, self.xr)), self.W_abstract.T)

    def predict_classical(self, u, du):
        return self.interval_dot(np.hstack((1, u, self.xc)), np.hstack((0, du, self.xr)), self.W_classical.T)

    def reset(self):
        self.xc = np.zeros((self.size,))
        self.xr = np.zeros((self.size,))