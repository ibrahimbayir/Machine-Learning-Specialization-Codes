import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu,linear
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

from public_tests_a1 import *

tf.keras.backend.set_floatx('float64')
from assigment_utils import *

tf.autograph.set_verbosity(0)

# Generate some data
X,y,x_ideal,y_ideal = gen_data(18, 2, 0.7)
print("X.shape", X.shape, "y.shape", y.shape)

#split the data using sklearn routine
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=1)
print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

fig, ax = plt.subplots(1,1,figsize=(4,4))
ax.plot(x_ideal, y_ideal, "--", color = "orangered", label="y_ideal", lw=1)
ax.set_title("Training, Test",fontsize = 14)
ax.set_xlabel("x")
ax.set_ylabel("y")

ax.scatter(X_train, y_train, color = "red",           label="train")
ax.scatter(X_test, y_test,   color = dlc["dlblue"],   label="test")
ax.legend(loc='upper left')
plt.show()


def eval_mse(y, yhat):
    """
    Calculate the mean squared error on a data set.
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:
      err: (scalar)
    """
    m = len(y)
    err = 0.0
    for i in range(m):
        ### START CODE HERE ###
        err += (y[i] - yhat[i]) ** 2
    ### END CODE HERE ###
    err /= 2 * m
    return (err)

    y_hat = np.array([2.4, 4.2])
    y_tmp = np.array([2.3, 4.1])
    eval_mse(y_hat, y_tmp)

    # BEGIN UNIT TEST
    test_eval_mse(eval_mse)
    # END UNIT TEST

    # create a model in sklearn, train on training data
    degree = 10
    lmodel = lin_model(degree)
    lmodel.fit(X_train, y_train)

    # predict on training data, find training error
    yhat = lmodel.predict(X_train)
    err_train = lmodel.mse(y_train, yhat)

    # predict on test data, find error
    yhat = lmodel.predict(X_test)
    err_test = lmodel.mse(y_test, yhat)

    print(f"training err {err_train:0.2f}, test err {err_test:0.2f}")

    # plot predictions over data range
    x = np.linspace(0, int(X.max()), 100)  # predict values for plot
    y_pred = lmodel.predict(x).reshape(-1, 1)

    plt_train_test(X_train, y_train, X_test, y_test, x, y_pred, x_ideal, y_ideal, degree)

    # Generate  data
    X, y, x_ideal, y_ideal = gen_data(40, 5, 0.7)
    print("X.shape", X.shape, "y.shape", y.shape)

    # split the data using sklearn routine
    X_train, X_, y_train, y_ = train_test_split(X, y, test_size=0.40, random_state=1)
    X_cv, X_test, y_cv, y_test = train_test_split(X_, y_, test_size=0.50, random_state=1)
    print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
    print("X_cv.shape", X_cv.shape, "y_cv.shape", y_cv.shape)
    print("X_test.shape", X_test.shape, "y_test.shape", y_test.shape)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.set_title("Training, CV, Test", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(X_train, y_train, color="red", label="train")
    ax.scatter(X_cv, y_cv, color=dlc["dlorange"], label="cv")
    ax.scatter(X_test, y_test, color=dlc["dlblue"], label="test")
    ax.legend(loc='upper left')
    plt.show()