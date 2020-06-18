from abstract_esn import AbstractESN
import numpy as np
from pathlib import Path

path = Path('./results/santafe/noisy')

def mean_squared_error(y_true, y_pred):
    try:
        return np.mean(np.abs((y_true - y_pred)**2))
    except:
        return -1

def mean_absolute_percentage_error(y_true, y_pred):
    try:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    except:
        return -1

if __name__ == '__main__':
    path.mkdir(parents=True, exist_ok=True)

    with open('dataset/SantaFe.D.txt', 'r') as f:
        train = list(map(float, f.readlines()))

    with open('dataset/SantaFe.D.cont.txt', 'r') as f:
        test = list(map(float, f.readlines()))

    n_neurons = 100
    n_inputs = 1
    connectivity = 0.2
    washout = 1000
    seed = 42
    a = 0.8
    spectral_radius = 0.1

    np.random.seed(42)

    with open(path/'results_closed.csv', 'w') as f:
        print('amplitude', 'n', 'type', 'MSE', 'MAPE', file=f,  sep=',')

    for du in np.arange(0, 1.0, 0.01):
        esn_abstract = AbstractESN(n_neurons, n_inputs, 1, connectivity, washout, seed, a, spectral_radius)
        esn_classical = AbstractESN(n_neurons, n_inputs, 1, connectivity, washout, seed, a, spectral_radius)

        X = np.zeros((1 + n_inputs + n_neurons, len(train) - washout - 1))
        dX = np.zeros((1 + n_inputs + n_neurons, len(train) - washout - 1))

        for i, u in enumerate(train[:-1]):
            (x, dx) = esn_abstract.step(u, du)
            if i >= washout:
                X[:, i - washout] = np.hstack((1, u, x))
                dX[:, i - washout] = np.hstack((1, du, dx))

        esn_classical.calculate_classical_weights(train[1 + washout:], X.T)
        esn_abstract.calculate_abstract_weights(train[1 + washout:], du, X.T, dX.T)

        x, _ = esn_abstract.step(train[-1], du)

        esn_classical.xc = np.copy(esn_abstract.xc)

        x_pretest = np.copy(x)

        for n in range(100):
            esn_classical.xc = np.copy(x_pretest)
            esn_abstract.xc = np.copy(x_pretest)
            Y_abstract = np.zeros((len(test) - 1, 1))
            Y_classical = np.zeros((len(test) - 1, 1))

            u_classical = test[0]
            u_abstract = test[0]
            for i in range(len(test[:-1])):
                noise = (np.random.rand() - 0.5) * du
                esn_classical.step(u_classical + noise, 0)
                esn_abstract.step(u_abstract + noise, 0)
                Y_classical[i, ...], _ = esn_classical.predict_classical(u_classical + noise, 0)
                Y_abstract[i, ...], _ = esn_abstract.predict_abstract(u_abstract + noise, 0)
                u_classical = Y_classical[i, ...]
                u_abstract = Y_abstract[i, ...]

            with open(path/'results_closed.csv', 'a') as f:
                print(du, n, 'abstract', mean_squared_error(test[1:], Y_abstract), mean_absolute_percentage_error(test[1:], Y_abstract), file=f, sep=',')
                print(du, n, 'classical', mean_squared_error(test[1:], Y_classical), mean_absolute_percentage_error(test[1:], Y_classical), file=f, sep=',')