import numpy as np
import scipy.linalg as scl
from sklearn.linear_model import Lasso

class My_Linear_Regression:
	def __init__(self, X_training, X_test, z, lambda_):
		self.X_training = X_training
		self.X_test = X_test
		self.z = z
		self.lambda_ = lambda_	

		"""
		###### Variables #######

		X_training = training data used to fit the model, (n, p) array 
		where n = number of training samples and p = number of features

		X_test = data to test the model on, (n, p) array
		where n = number of test samples
		
		z = the response, (n, ) 
		lambda = the penalty, float
		"""

	def My_OLS(self):

		"""
		Ordinary Least Square method

		######  Method   #######
	
		Solves the OLS with SVD and Moore Penrose psudoinverse
		omega = X^Ty/(X^TX)
	
		"""

		# Calculate the Ordinary Least Square             
				
		U, D, VT = np.linalg.svd(self.X_training)
		sigma = np.zeros((U.shape[0], VT.shape[1]))
		np.fill_diagonal(sigma, D)
		self.beta = VT.T.dot(np.linalg.pinv(sigma)).dot(U.T).dot(self.z)
	
	def My_Ridge(self):

		"""
		Ridge regression method

		######  Method   #######
	
		Solves the ridge regression
		omega = X^Ty/(X^TX + lambda)
		"""
		
		# Calculate the Ridge regression

		m = self.X_training.shape[1]
		self.beta = self.X_training.T.dot(self.z).dot(np.linalg.inv(self.X_training.T.dot(self.X_training) + np.eye(m)*self.lambda_))

	def My_Lasso(self):

		"""
		Lasso regression method

		Calculate the Lasso regression using scikit learn
		"""
		
		# Calculate the Lasso regression

		lasso = Lasso(self.lambda_)
		self.fit = lasso.fit(self.X_training, self.z)
		self.beta = lasso.coef_

	def My_Predict(self, X_test, l):

		if l == True:
			z_predict = self.fit.predict(X_test)
	
		else:
			z_predict = X_test.dot(self.beta)

		return z_predict

	def My_Beta(self):

		return self.beta

		

             

