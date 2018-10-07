from sklearn.linear_model import Ridge, Lasso
import numpy as np

class My_Linear_Regression:
	def __init__(self, X_training, X_test, z, lambda_):
		self.X_training = X_training
		self.X_test = X_test
		self.z = z
		self.lambda_ = lambda_	

		"""
		Check if the data sample is split in to training
		and test data. If not, then the two variables are 		
		the same and we do at fit and predict on the same
		data.
		"""
		#if X_test == None:
		#	self.X_test = self.X_training	

	def My_OLS(self):

		"""
		Ordinary Least Square method
	 	
		###### Variables #######

		method = the selected regression method to use
		alpha = lambda in Ridge or Lasso regression 
	
		######  Method   #######
	
		Solves the OLS ....
		up to fifth order
	
		"""

		# Calculate the Ordinary Least Square             
		self.beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z)

	def My_Ridge(self):

		"""
		Ridge regression method
		"""
		n = np.size(self.X_training,0)   # size of column (number of rows)
		m = np.size(self.X_training,1)   # number of columns
		# Separate the first column in X and first value in z
		X_new = self.X_training[:,1:]

		# Calculate mean of z_training equals beta_0
		beta_0 = 1.0/(n)*sum(self.z)
		#I = np.identity(m-1)
		#I_lambda = I*self.lambda_ 
		#X_ridge = X_new + I_lambda
		X_ridge = self.lambda_*X_new
		
		# Calculate the Ridge regression
		self.beta = np.linalg.inv(X_ridge.T.dot(X_ridge)).dot(X_ridge.T).dot(self.z)
		self.beta = np.insert(self.beta, 0, beta_0)

	def My_Lasso(self):

		# Calculate the Lasso regression using scikit learn
		
		lasso = Lasso(self.lambda_)
		self.beta = lasso.fit(self.X_training, self.z)

	def My_Predict(self, X_test, l):

		if l == True:
			z_predict = self.beta.predict(X_test)
	
		else:
			z_predict = X_test.dot(self.beta)

		return z_predict

		

             

