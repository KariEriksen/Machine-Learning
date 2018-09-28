import numpy as np
from numpy.random import randint
from linear_regression import My_Linear_Regression 

class Bootstrap:
    def __init__(self, X, B, alpha, split):
        self.X = X
	self.B = B
	self.alpha = alpha
	self.split = split
	self.method = method
	

    def My_Bootstrap(self):

	"""
	Bootstrap class
 	
	###### Variables #######

	data is an array containing all data from experiment
	B = number of bootstraps
	split = amount split to training data in decimal percentage
	
	######  Method   #######
	
	Resampling method with replacement, takes data sample
	returns the standard deviation and mean of the resampled
	
	Samples values with replacement and calculates the mean 
	Repeated n times
	Generates sampling distribution of mean
	"""
	# Split the data according to the given decimal percentage
	n = np.size(X,0)   # size of column (number of rows)
	C = n*split
	X_test = np.random.choice(x, C))
	
	np.random.shuffle(X)
	training, test = X[:C,:], X[C:,:]
	
	# Do linear regression
	lr = My_Linear_Regression(X, alpha)

	# Ordinary Least Square method
	if method == 'OLS':
		z_predict = lr.My_OLS

	# Ridge regression
	elif method == 'Ridge':
		z_predict = lr.My_Ridge

	#Lasso regression
	elif method == 'Lasso':
		z_predict = lr.My_Lasso

	else:
		print('You have forgotten to select method; OLS, Ridge or Lasso.')
	

	n = len(data)
	t = zeros(n)
        for i in range(n):
		sample = data[randint(0,n,n)]
        	t[i] = mean(sample)
	
	# Calculate the mean and standard deviation
	mean = 1.0/n*(sum(t))  
	std = 1.0/n*(sum())

	# Calculate the Mean Square Error
	diff = z - z_predict
	MSE = 1/(n*n)*(sum(diff*diff))
	
	return std, mean

