# %%  imports
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# %% Part 1

iris = pd.read_csv("iris copy.data", names=["x1" , "x2" , "x3" , "x4" , "y"])
X = iris[["x1" , "x2" , "x3" , "x4"]]
y = iris[["y"]]

def calc_r2(X, y):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    return r2_score(y, y_pred)

for i in range(1,4):
    r2 = calc_r2(X.iloc[:, [0]], X.iloc[:, [i]])
    print(f"R2 score betwwen x1 and x{i+1} is {r2}")

# %% Implement gradient descent

def sq_errors(y_true, y_pred):
    return float((1/len(y_pred)) * sum((y_true - y_pred) ** 2))

def accuracy(X, y, w, fn):
    y_pred = predict(X, w, fn)
    correct = sum(np.where(y == y_pred, 1, 0))
    n = len(y)
    return float(correct / n)

def calc_accuracy(X, y, w):
    return sum(np.where(X.dot(w) > 0, 1, -1) == y) / len(y)

def predict(X, w, fn=None):
    if fn: 
        return fn(X @ w)
    return X @ w

def step(z):
    return np.where(z >= 0, 1, -1)

def calculate_gradient(X, y, w, fn=None):
    y_pred = predict(X, w, fn)
    return (-1.0/len(y)) * (np.transpose(X) @ (y - y_pred))

def gradient_descent(X, y, fn=None, learn_rate=0.01, max_iter=100, accuracy=0.001):
    xcols = X.shape[1]
    w = np.random.random(xcols).reshape([X.shape[1], 1]) # random starting vector size = number of columns
    for i in range(max_iter):
        w -= learn_rate * calculate_gradient(X, y, w, fn)
        if i % 10 == 0:
            y_p = predict(X, w, fn)
            print(f"MSE: {sq_errors(y, y_p):.3f}")
            a = calc_accuracy(X, y, w)
            # print(f"MSE: {sq_errors(y, y_p):.3f}")
            # a2 = accuracy(X,y,w,fn)  # 'float' object is not callable
            print(f"Accuracy: {float(a):.3f}")
            print(f"{w=}")

    return w




# %%
iris_12 = pd.read_csv("iris_12.data")
X12 = iris_12.iloc[:, 0:4]
y12 = iris_12.iloc[:, [4]]
y12 = np.where(y12 == 1, 1 , -1)
X12_train, X12_test, y12_train, y12_test = train_test_split(X12, y12, train_size=0.5, random_state=55)

w12 = gradient_descent(X12_train, y12_train, step)

# %%


print(f"{accuracy(X12_test, y12_test, w12, step)}")

# %%
