from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def LR_learn(x, y, nb_iteration, epsilon, eta):
    sts = StandardScaler()
    norm_sts = sts.fit(x)
    
    x = norm_sts.transform(x)
    x = np.c_[np.ones(len(x)), x]

    w = np.zeros(len(x[0]))

    cpt = 0
    diff_error = 10**5
    e_back = 10**5

    while cpt < nb_iteration and diff_error > epsilon:
        cpt = cpt + 1
        h = np.dot(x, w)
        r = y - h
        f = r**2
        e = 0.5 * np.mean(f)
        grad_f = -2 * np.multiply(x.transpose(), r)
        grad_e = 0.5 * np.mean(grad_f, axis=1)
        w = w - eta * grad_e
        diff_error = np.abs(e - e_back)
        e_back = e

    print('Number of iterations : ' + str(cpt))
    return w, norm_sts


def LR_predict(x, w, norm_sts):
    x = norm_sts.transform(x)
    x = np.c_[np.ones(len(x)), x]
    return np.dot(x, w)


boston = datasets.load_boston()
x = boston.data
y = boston.target

x_tr = x[0::2, :]
y_tr = y[0::2]

x_ts = x[1::2, :]
y_ts = y[1::2]

nb_iteration = 100
eta = 0.1
epsilon = 0.1

w, norm_sts = LR_learn(x_tr, y_tr, nb_iteration, epsilon, eta)
predicted = LR_predict(x_ts, w, norm_sts)

mse = np.mean((predicted - y_ts)**2)
print('MSE: ' + str(mse))

fig, ax = plt.subplots()
ax.scatter(y_ts, predicted, edgecolors=(0, 0, 0))
ax.plot([y_ts.min(), y_ts.max()], [y_ts.min(), y_ts.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
