from sklearn import datasets
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_result(x, p, index, r):
    figure, axs = plt.subplots(2, 5)
    figure.canvas.set_window_title('Classification algorithms')
    for i in range(2):
        for j in range(5-i):
            title = "%s (%s)" % (index[i*5+j], r[i*5+j]["score_prediction"])
            axs[i, j].scatter(x[:, 0], x[:, 1], c=p[i*5+j])
            axs[i, j].set_title(title, fontsize=10)
    plt.show()


def display_result(result):
    df = pd.DataFrame(result, index=index)
    pd.set_option('display.max_columns', 0)

    print(df)


def process_data(x, y):
    x_tr = x[0::2, :]
    y_tr = y[0::2]
    x_ts = x[1::2, :]
    y_ts = y[1::2]

    return x_tr, x_ts, y_tr, y_ts


def process_classification(m, x, y):
    x_tr, x_ts, y_tr, y_ts = process_data(x, y)

    model = m
    model = model.fit(x_tr, y_tr)

    nb_sample_training = len(x_tr)
    nb_sample_prediction = len(x_ts)
    score_training = str(round(model.score(x_tr, y_tr) * 100, 3)) + '%'
    predicted = model.predict(x_ts)
    nb_error = np.sum(predicted != y_ts)
    score_prediction = str(round(metrics.accuracy_score(y_ts, predicted) * 100, 3)) + '%'
    confusion_matrix = metrics.confusion_matrix(y_ts, predicted)
    result = {'score_training': score_training,
              'score_prediction': score_prediction,
              'nb_error': nb_error,
              'confusion_matrix': confusion_matrix,
              'nb_sample_prediction': nb_sample_prediction,
              'nb_sample_training': nb_sample_training}

    return x_ts, predicted, result


if __name__ == '__main__':
    breast_cancer = datasets.load_breast_cancer()

    x = breast_cancer.data
    y = breast_cancer.target

    predicted_array = []
    result_array = []

    modelsList = [LogisticRegression(),
                  KNeighborsClassifier(),
                  SVC(),
                  LinearSVC(),
                  NuSVC(),
                  GaussianNB(),
                  DecisionTreeClassifier(),
                  LinearDiscriminantAnalysis(),
                  QuadraticDiscriminantAnalysis()]

    index = ['LogisticRegression',
             'KNeighborsClassifier',
             'SVC',
             'LinearSVC',
             'NuSVC',
             'GaussianNB',
             'DecisionTreeClassifier',
             'LinearDiscriminantAnalysis',
             'QuadraticDiscriminantAnalysis']

    for model in modelsList:
        x_ts, predicted, result = process_classification(model, x, y)
        predicted_array.append(predicted)
        result_array.append(result)

    display_result(result_array)
    plot_result(x_ts, predicted_array, index, result_array)
