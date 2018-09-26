import numpy as np
from numpy.random import randint

class Bootstrap:
    def __init__(self, data):
        self.data = data

    def bootstrap(self, B, split):

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

	
	n = len(data)
	t = zeros(n)
        	for i in range(n):
		sample = data[randint(0,n,n)]
        		t[i] = mean(sample)
	
	mean = 1.0/n*(sum(t))  
	std = 1.0/n*(sum())
	
	return std, mean

