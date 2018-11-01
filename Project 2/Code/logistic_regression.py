import numpy as np
import scipy.linalg as scl
from math import exp, log

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
		beta = the weights of the regression
		lambda = the penalty
		"""

	def sigmoid(self, t):

		"""
		Returns the sigmoid function f(s) = 1/(1 + e^(-s))
 
		It represents the probability of a data point being 
		in a certain category. In the case of the Ising model 
		this corresponds to the configuration being in one 
		certain state, ordered or disordered.
		
		Note: 1 - f(s) = f(-s) 
		      s = X.T.dot(b)
		"""
		sigm = (1.0 + np.exp(-t))**(-1)
		return sigm

	def cross_entropy(self, w):
		
		"""
		Calculates the cost function for the logistic 
		regression case, i.e. the cross entropy. 
		"""
		
		n = np.size(self.X_training,0) # size of row (number of columns) 
		s_i = self.X_training.dot(w)
		f = self.sigmoid(s_i)
		C = -self.z*np.log(f) - (1 - self.z)*np.log(1 - f)
		return np.sum(C)

	def deri_cross_entropy(self, w):
	 	
		"""
		Calculates the derivative of the cross entropy.
		
		To be used in the gradient descent method in order
		to minimize the cost function, i.e. finding the 
		local minima. 

		Returns a n x m matrix, where m = (L**2), containing
		the derivatives of each spin-element (in the rows) for all 
		configurations (in the columns).
		"""
		n = np.size(self.X_training,0) # number of samples
		s_i = self.X_training.dot(w)
		f = self.sigmoid(s_i)
		c = f - self.z
		dC = c.dot(self.X_training)
		#print (dC)
		#exit()
		#print (c.shape, dC.shape)
		#exit() 
		return dC



