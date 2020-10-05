# %%  imports
import numpy as np
import pandas as pd 
from sklearn.preprocessing import minmax_scale, OneHotEncoder
from sklearn.model_selection import train_test_split
import model_multiclass as lr # use logistic regression algorith from lectures


# %% pre-processing

columnnames = ["ClassID"
        ,"Alcohol"
        ,"Malic acid"
        ,"Ash"
        ,"Alcalinity of ash"
        ,"Magnesium"
        ,"Total phenols"
        ,"Flavanoids"
        ,"Nonflavanoid phenols"
        ,"Proanthocyanins"
        ,"Color intensity"
        ,"Hue"
        ,"OD280/OD315 of diluted wines"
        ,"Proline"
        ]
wine = pd.read_csv("wine.data", header=None, names=columnnames)
X_raw = wine.drop(columns="ClassID")
y = wine.iloc[:, 0]

#normalise input between 0 and 1
X = pd.DataFrame(minmax_scale(X_raw), columns=columnnames[1:])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)

# one hot encoding
y_train_onehot = [[1 if j == i else 0 for j in range(3)] for i in y_train]
y_test_onehot = [[1 if j == i else 0 for j in range(3)] for i in y_test]

gd = []
sgd = []


# train_data = X_train.join(y_train).reset_index(drop=True)
# test_data = X_test.join(y_test).reset_index(drop=True)
# num_features = len(X_train.columns)
# %%

model = lr.logistic_regression(1000, X_train, X_test, y_train, y_test, 0.1)
model.SGD()    
# for i in range(10):

# %%


# %%
