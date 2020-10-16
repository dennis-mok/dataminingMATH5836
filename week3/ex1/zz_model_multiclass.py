 
 # by R. Chandra
 #Source: https://github.com/rohitash-chandra/logisticregression_multiclass



from math import exp
import numpy as np
import random

SIGMOID = 1
STEP = 2
LINEAR = 3

 

class logistic_regression:

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
		self.use_activation = SIGMOID #SIGMOID # 1 is  sigmoid , 2 is step, 3 is linear 
		self.out_delta = np.zeros(self.num_outputs)

		print(f"{self.w=}") 
		print(f"{self.b=}") 

	def reset_model(self):
		self.w = np.random.uniform(-0.5, 0.5, (self.num_features, self.num_outputs))  
		self.b = np.random.uniform(-0.5, 0.5, self.num_outputs) 
 
	def activation_func(self,z_vec):
		if self.use_activation == SIGMOID:
			y =  1 / (1 + np.exp(-z_vec)) # sigmoid/logistic
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

	def predict_(self, x_vec ):  # implementation using for loops
		weightsum = 0
		output = np.zeros(self.num_outputs)

		for y in range(0, self.num_outputs):
			for x in range(0, self.num_features): 
				weightsum   +=    x_vec[x] * self.w[x,y] 
			output[y] = self.activation_func(weightsum- self.b[y])
			weightsum  = 0 

		return output
	
	def gradient(self, x_vec, output, actual):   
		if self.use_activation == SIGMOID :
			out_delta =  -(actual - output)*(output*(1-output)) 
		else: # for linear and step function  
			out_delta =  -(actual - output) 
		return out_delta

	def update(self, x_vec, output, actual):   # implementation using dot product 

		x_vec = np.reshape(x_vec, (-1, 1))  # reshapes 1D array as Nx1D array for numpy dot operation. 
		out_delta= np.reshape(self.out_delta, (-1, 1))

		self.w +=  x_vec.dot(out_delta.T) * self.learn_rate
		self.b +=  (1 * self.learn_rate * self.out_delta)

	def update_(self, x_vec, output, actual): # implementation using for loops 

		for x in range(0, self.num_features):
			for y in range(0, self.num_outputs):
				self.w[x,y] += self.learn_rate * self.out_delta[y] * x_vec[x] 

		for y in range(0, self.num_outputs):
			self.b += -1 * self.learn_rate * self.out_delta[y]

 

	def squared_error(self, prediction, actual):
		return  np.sum(np.square(prediction - actual))/prediction.shape[0]# to cater more in one output/class

	def test_model(self, X_data, y_data, tolerance):  

		num_instances = len(y_data)

		class_perf = 0
		sum_sqer = 0   
		for s in range(0, num_instances):	

			prediction = self.predict(X_data[s]) 
			sum_sqer += self.squared_error(prediction, y_data)

			pred_binary = np.where(prediction > (1 - tolerance), 1, 0)

			# print(s, y_data, prediction, pred_binary, sum_sqer, ' s, y_data, prediction, sum_sqerr')

 

			if( (y_data==pred_binary).all()):
				class_perf =  class_perf + 1   

		rmse = np.sqrt(sum_sqer/num_instances)

		percentage_correct = float(class_perf)/num_instances * 100 

		print(f"{rmse=}, {percentage_correct=}")
		# note RMSE is not a good measure for multi-class probs

		return ( rmse, percentage_correct)



 
	def SGD(self):   
		
			epoch = 0 
			shuffle = True

			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					if shuffle ==True:
						i = random.randint(0, self.num_train-1)


					input_instance  =  self.X_train[i]  # train_data[i,0:self.num_features]  
					actual  = self.y_train[i]  # self.train_data[i,self.num_features:]  
					prediction = self.predict(input_instance) 
					sum_sqer += self.squared_error(prediction, actual)
					self.out_delta = self.gradient( input_instance, prediction, actual)    # major difference when compared to GD
					#print(input_instance, prediction, actual, s, sum_sqer)
					self.update(input_instance, prediction, actual)

			
				#print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.X_train, self.y_train, 0.3) 
			# rmse_test =0
			# test_perc =0
			rmse_test, test_perc = self.test_model(self.X_train, self.y_train, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				

	def GD(self):   
		
			epoch = 0 
			while  epoch < self.max_epoch:
				sum_sqer = 0
				for s in range(0, self.num_train): 
					input_instance  =  self.X_train[s] # self.train_data[s,0:self.num_features]  
					actual  = self.y_train[s]
					prediction = self.predict_(input_instance) 
					sum_sqer += self.squared_error(prediction, actual) 
					self.out_delta += self.gradient( input_instance, prediction, actual)    # this is major difference when compared with SGD

					#print(input_instance, prediction, actual, s, sum_sqer)
				self.update(input_instance, prediction, actual)

			
				#print(epoch, sum_sqer, self.w, self.b)
				epoch=epoch+1  

			rmse_train, train_perc = self.test_model(self.X_train, self.y_train, 0.3) 
			# rmse_test =0
			# test_perc =0
			rmse_test, test_perc = self.test_model(self.X_train, self.y_train, 0.3)
  
			return (train_perc, test_perc, rmse_train, rmse_test) 
				
	
 

#------------------------------------------------------------------
#MAIN


def main(): 

	random.seed()
	 
	dataset = [[2.7810836,2.550537003,0], # sample binary classification data
		[1.465489372,2.362125076,0],
		[3.396561688,4.400293529,0],
		[1.38807019,1.850220317,0],
		[3.06407232,3.005305973,0],
		[7.627531214,2.759262235,1],
		[5.332441248,2.088626775,1],
		[6.922596716,1.77106367,1],
		[8.675418651,-0.242068655,1],
		[7.673756466,3.508563011,1]]


	train_data = np.asarray(dataset) # convert list data to numpy
	test_data = train_data


	dataset_onehot = [[2.7810836,2.550537003,0, 1], # binary classification with one-hot encoding 
		[1.465489372,2.362125076,0, 1],
		[3.396561688,4.400293529,0, 1],
		[1.38807019,1.850220317,0, 1],
		[3.06407232,3.005305973,0, 1],
		[7.627531214,2.759262235,1, 0],
		[5.332441248,2.088626775,1, 0],
		[6.922596716,1.77106367,1, 0],
		[8.675418651,-0.242068655,1, 0],
		[7.673756466,3.508563011,1, 0]]


	train_data_onehot = np.asarray(dataset_onehot) # convert list data to numpy
	test_data_onehot = train_data

	learn_rate = 0.3
	num_features = 2
	num_epocs = 20

	print(train_data)
	 

	#lreg = logistic_regression(num_epocs, train_data, test_data, num_features, learn_rate)
	#(train_perc, test_perc, rmse_train, rmse_test) = lreg.SGD()
	#(train_perc, test_perc, rmse_train, rmse_test) = lreg.GD() 

	#-------------------------------

	# lreg_sgd = logistic_regression(num_epocs, train_data_onehot, test_data_onehot, num_features, learn_rate)
	# (train_perc, test_perc, rmse_train, rmse_test) = lreg_sgd.SGD()

	# lreg_gd = logistic_regression(num_epocs, train_data_onehot, test_data_onehot, num_features, learn_rate)
	# (train_perc, test_perc, rmse_train, rmse_test) = lreg_gd.GD() 

	 
	 
	# Iris data (3 classes)
	#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/ 
	# using first 5 samples from each class with one-hot encoding 

	iris_train = [[5.1,3.5,1.4,0.2, 1, 0, 0], 
				[4.9,3.0,1.4,0.2, 1, 0, 0],
				[4.7,3.2,1.3,0.2,1, 0, 0],
				[4.6,3.1,1.5,0.2, 1, 0, 0],
				[5.0,3.6,1.4,0.2,1, 0, 0],
				[7.0,3.2,4.7,1.4,0, 1, 0],
				[6.4,3.2,4.5,1.5,0, 1, 0],
				[6.9,3.1,4.9,1.5,0, 1, 0],
				[5.5,2.3,4.0,1.3,0, 1, 0],
				[6.5,2.8,4.6,1.5,0, 1, 0],
				[6.3,3.3,6.0,2.5,0, 0, 1],
				[5.8,2.7,5.1,1.9,0, 0, 1],
				[7.1,3.0,5.9,2.1,0, 0, 1],
				[6.3,2.9,5.6,1.8,0, 0, 1],
				[6.5,3.0,5.8,2.2,0, 0, 1]]


	train_data = np.asarray(iris_train) # convert list data to numpy
	test_data = train_data  # assume test is same as train (you can change by reading data from file or create train test split)


	print(train_data, ' iris data')

	learn_rate = 0.1
	num_features = 4
	num_epocs = 50

	 

	# lreg_sgd = logistic_regression(num_epocs, train_data, test_data, num_features, learn_rate)
	# (train_perc, test_perc, rmse_train, rmse_test) = lreg_sgd.SGD()

	# lreg_gd = logistic_regression(num_epocs, train_data, test_data, num_features, learn_rate)
	# (train_perc, test_perc, rmse_train, rmse_test) = lreg_gd.GD() 


if __name__ == "__main__": main()