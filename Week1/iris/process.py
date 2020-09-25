# Use simple find and replace to convert the class labels to 1, 2, and 3 in the dataset. 
# iris copy.data

# %% Read the data and report mean and standard deviation for each column in the features (4 features)
import numpy as np
import pandas as pd
import random

iris_data = pd.read_csv("iris copy.data", names=["x1" , "x2" , "x3" , "x4" , "y"])

# for col in iris_data.columns:
#     print(f"Mean of {col} = {np.mean(iris_data[col])}")
#     print(f"StdDev of {col} = {np.std(iris_data[col])}")

print(iris_data.mean())
print(iris_data.std())

# %% Report the class distribution (i. e number of instances for each class)
iris_data.groupby(["y"]).size()    

# %% Show histogram for each feature. Note you need to use a single function/method that 
# outputs the histogram with a given filename. eg. feature1.png which is given as a parameter 
# to the function. A for loop should be used to call the function/method
h = iris_data.hist()

# %% Split data into a train and test test. Use 60 percent data in the training and test set 
# which is assigned i. randomly ii. assigned by first 60 percent as train and rest as test. 
train = iris_data.sample(frac=0.6, random_state=1)
test = iris_data.drop(train.index)

# Use previous functions to report the mean and standard deviation of the train and test set 
# and class distribution and also the histograms for each feature. 
print(train.mean())
print(train.std())
train.hist()

print(test.mean())
print(test.std())
test.hist()

# Create another subset of the train and test set where only 1 feature selected by the user 
# makes the dataset with the class. 


# Create a subset of the dataset where you consider only instances that feature class 1 or 2, 
# so that you treat this problem as a binary classification problem later, 
# i.e save it as binary_iristrain.txt and binary_iristest.txt. 
# Carry out the stats and visuals in Step 6 for this dataset. 

# Can you normalise the input features between [0 and 1] ? 
# Write code that can do so and save normalised versions.


# %%
