import numpy as np
import scipy.linalg as scl

class Logistic_Regression:
	def __init__(self, X_training, X_test, z, lambda_):
		self.X_training = X_training
		self.X_test = X_test
		self.z = z
		self.lambda_ = lambda_	

		"""
		###### Variables #######

		X_training = training data used to fit the model
		X_test = data to test the model on
		z = the response
		lambda = the penalty
		"""

	def sigmoid(self):

		"""
		Returns the sigmoid function f(s) = 1/(1 + e^(-s))

		In the case of the Ising model this corresponds to 
		the probability of a data point being in a certain 	
		category. 
		
		Note: 1 - f(s) = f(-s) 
		"""

		return 0

	def cross_entropy(self):

		"""
		Calculates the cost function for the logistic 
		regression case, i.e. the cross entropy. 
		"""

		return 0

	def deri_cross_entropy(self):
		
		"""
		Calculates the derivative of the cross entropy.
		
		To be used in the gradient descent method in order
		to minimize the cost function, i.e. finding the 
		local minima. 
		"""

		return 0
