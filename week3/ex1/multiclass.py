from math import exp
import numpy as np
from random import random

SIGMOID = 1
STEP = 2
LINEAR = 3

class logistic_regression():

	def __init__(self, num_epocs, X_train, X_test, y_train, y_test, learn_rate):
		# self.train_data = train_data
		# self.test_data = test_data 

		self.X_train = np.array(X_train)
		self.X_test = np.array(X_test)
		self.y_train = np.array(y_train)	
		self.y_test = np.array(y_test)	
		self.num_features = self.X_train.shape[1]
		self.num_outputs = self.y_train.shape[1]
		self.num_train = len(self.X_train)

		print(self.num_features, self.num_outputs)
		#self.w = np.random.uniform(-0.5, 0.5, num_features)  # in case one output class
		self.w = np.random.uniform(-0.5, 0.5, (self.num_features, self.num_outputs))  
		self.b = np.random.uniform(-0.5, 0.5, self.num_outputs) 
		self.learn_rate = learn_rate
		self.max_epoch = num_epocs
		self.use_activation = SIGMOID # SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear 
		self.out_delta = np.zeros(self.num_outputs)

		print(f"{self.w=}") 
		print(f"{self.b=}") 

	def reset_model(self):
		self.w = np.random.uniform(-0.5, 0.5, (self.num_features, self.num_outputs))  
		self.b = np.random.uniform(-0.5, 0.5, self.num_outputs) 
 

	def activation_func(self, z_vec):
		if self.use_activation == SIGMOID:
			y =  1.0 / (1 + exp(-z_vec)) # sigmoid/logistic
		elif self.use_activation == STEP:
			y = (z_vec > 0).astype(int) # if greater than 0, use 1, else 0
			#https://stackoverflow.com/questions/32726701/convert-real-valued-numpy-array-to-binary-array-by-sign
		else:
			y = z_vec
		return y
    
	def predict(self, x_vec ): # implementation using dot product
		z_vec = x_vec.dot(self.w) - self.b 
		output = self.activation_func(z_vec) # Output  
		return output

	def gradient(self, x_vec, output, actual):   
		if self.use_activation == SIGMOID :
			out_delta =   (output - actual)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =   (output - actual) 
		return out_delta
    