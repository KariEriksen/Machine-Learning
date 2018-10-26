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

	def sigmoid(self, X, b):

		"""
		Returns the sigmoid function f(s) = 1/(1 + e^(-s))
 
		It represents the probability of a data point being 
		in a certain category. In the case of the Ising model 
		this corresponds to the configuration being in one 
		certain state, ordered or disordered.
		
		Note: 1 - f(s) = f(-s) 
		      s = X.T.dot(b)
		"""

		sigm = 1.0/(1 + exp(-X.T.dot(b)))		
		return sigm

	def deri_cross_entropy(self, beta):
		
		"""
		Calculates the derivative of the cross entropy.
		
		To be used in the gradient descent method in order
		to minimize the cost function, i.e. finding the 
		local minima. 

		Returns a n x m matrix, where m = (L**2), containing
		the derivatives of each spin-element (in the rows) for all 
		configurations (in the columns).
		"""
	
		n = np.size(self.X_training,1) # size of row (number of columns) 
		dC = np.zeros((n, n))
		for i in range(n):
			#print ('X = %s, beta = %s' % (self.X_training[i].shape, beta.shape))
			f_i = self.sigmoid(self.X_training[i,:], beta)
			y_i = self.z[i]
			x_i = self.X_training[i]
			#print ('f_i = %s, y_i = %s, x_i = %s' % (f_i, y_i, x_i))
			dC[i,:] += (f_i - y_i)*(x_i)
		return dC



