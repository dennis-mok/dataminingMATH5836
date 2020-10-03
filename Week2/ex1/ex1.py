# %% imports
import pandas as pd
import numpy as np
from random import sample
from sklearn.linear_model import LogisticRegression, LassoCV, RidgeCV
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error, roc_auc_score, 
        roc_curve, f1_score, precision_recall_curve)
import matplotlib.pyplot as plt

# %% load data 
columns = ["preg", "plas", "pres", "skin", "test", "mass", "pedi", "age", "class"]
# preg = Number of times pregnant
# plas = Plasma glucose concentration a 2 hours in an oral glucose tolerance test\
# pres = Diastolic blood pressure (mm Hg)
# skin = Triceps skin fold thickness (mm)
# test = 2-Hour serum insulin (mu U/ml)
# mass = Body mass index (weight in kg/(height in m)^2)
# pedi = Diabetes pedigree function
# age = Age (years)
# class = Class variable (1:tested positive for diabetes, 0: tested negative for diabetes)

data = pd.read_csv("pima-indians-diabetes.csv", names=columns)

# %% train model
X = data.drop(columns="class")
y = data[["class"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=123)

model = LogisticRegression()
model.fit(X_train, y_train)

# %% 



for xy in [(X_train, y_train, "Training"), (X_test, y_test, "Testing")]:
    # part 1
    print(xy[2])
    X_ = xy[0]
    y_true = xy[1]
    y_pred = model.predict(xy[0])
    y_predp = model.predict_proba(xy[0])[:, 1]
    print(f"Log loss: {log_loss(y_true, y_predp):.3f}")
    print(f"Accuracy: {model.score(X_, y_true):.3f}")
    print(f"RMSE: {mean_squared_error(y_true, y_predp, squared=False):.3f}")
    
    # part 2
    print(f"AUC score: {roc_auc_score(y_true, y_predp):.3f}")
    print(f"F1 score: {f1_score(y_true, y_pred):.3f}")
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_predp)

    plt.figure(1)  # roc auc curve
    plt.plot(lr_fpr, lr_tpr, marker=".", label=xy[2])

    plt.figure(2)  # precision recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_predp)
    plt.plot(recall, precision, label=xy[2])
    print("---\n")

plt.figure(1)
ns_pred = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No skill")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()

plt.figure(2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()
# %% class imbalanced

data2 = data.loc[data["class"] == 0]
data2 = data2.append(data.loc[data["class"] == 1].sample(frac=0.2))
X2 = data2.drop(columns="class")
y2 = data2[["class"]]

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, train_size=0.6, random_state=123)

model2 = LogisticRegression()
model2.fit(X2_train, y2_train)

# %% class imbalance


for xy in [(X2_train, y2_train, "Training"), (X2_test, y2_test, "Testing")]:
    # part 1
    print(xy[2])
    X_ = xy[0]
    y2_true = xy[1]
    y_pred = model2.predict(xy[0])
    y_predp = model2.predict_proba(xy[0])[:, 1]
    print(f"Log loss: {log_loss(y2_true, y_predp):.3f}")
    print(f"Accuracy: {model2.score(X_, y2_true):.3f}")
    print(f"RMSE: {mean_squared_error(y2_true, y_predp, squared=False):.3f}")
    
    # part 2
    print(f"AUC score: {roc_auc_score(y2_true, y_predp):.3f}")
    print(f"F1 score: {f1_score(y2_true, y_pred):.3f}")
    lr_fpr, lr_tpr, _ = roc_curve(y2_true, y_predp)

    plt.figure(3)  # roc auc curve
    plt.plot(lr_fpr, lr_tpr, marker=".", label=xy[2])

    plt.figure(4)  # precision recall curve
    precision, recall, _ = precision_recall_curve(y2_true, y_predp)
    plt.plot(recall, precision, label=xy[2])
    print("---\n")

plt.figure(3)
ns_pred = [0 for _ in range(len(y2_test))]
ns_fpr, ns_tpr, _ = roc_curve(y2_test, ns_pred)
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No skill")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()

plt.figure(4)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()


# %% cross validation

cv_scores = cross_validate(model, X_train, y_train, cv=10, 
        scoring=("neg_log_loss", "accuracy", "neg_mean_squared_error", "roc_auc"))

for sc in cv_scores:
    print(sc, np.mean(cv_scores[sc]))
# %%  lasso and ridge

lasso = LassoCV()
lasso.fit(X_train, y_train)
# lasso_pred = lasso.predict(X_test)


for xy in [(X_train, y_train, "Training"), (X_test, y_test, "Testing")]:
    # part 1
    print(xy[2])
    X_ = xy[0]
    y_true = xy[1]
    y_predp = lasso.predict(xy[0])
    y_pred = np.where(y_predp < 0.5, 0, 1)
    # y_predp = lasso.predict_proba(xy[0])[:, 1]
    print(f"Log loss: {log_loss(y_true, y_predp):.3f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_true, y_predp, squared=False):.3f}")
    
    # part 2
    print(f"AUC score: {roc_auc_score(y_true, y_predp):.3f}")
    print(f"F1 score: {f1_score(y_true, y_pred):.3f}")
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_predp)

    plt.figure(5)  # roc auc curve
    plt.plot(lr_fpr, lr_tpr, marker=".", label=xy[2])

    plt.figure(6)  # precision recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_predp)
    plt.plot(recall, precision, label=xy[2])
    print("---\n")

plt.figure(5)
ns_pred = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No skill")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()

plt.figure(6)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

# %%


ridge = RidgeCV()
ridge.fit(X_train, y_train)
# ridge_pred = ridge.predict(X_test)


for xy in [(X_train, y_train, "Training"), (X_test, y_test, "Testing")]:
    # part 1
    print(xy[2])
    X_ = xy[0]
    y_true = xy[1]
    y_predp = ridge.predict(xy[0])
    y_pred = np.where(y_predp < 0.5, 0, 1)
    # y_predp = ridge.predict_proba(xy[0])[:, 1]
    print(f"Log loss: {log_loss(y_true, y_predp):.3f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(f"RMSE: {mean_squared_error(y_true, y_predp, squared=False):.3f}")
    
    # part 2
    print(f"AUC score: {roc_auc_score(y_true, y_predp):.3f}")
    print(f"F1 score: {f1_score(y_true, y_pred):.3f}")
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_predp)

    plt.figure(7)  # roc auc curve
    plt.plot(lr_fpr, lr_tpr, marker=".", label=xy[2])

    plt.figure(8)  # precision recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_predp)
    plt.plot(recall, precision, label=xy[2])
    print("---\n")

plt.figure(7)
ns_pred = [0 for _ in range(len(y_test))]
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_pred)
plt.plot(ns_fpr, ns_tpr, linestyle="--", label="No skill")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.legend()

plt.figure(8)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.show()

# %%
