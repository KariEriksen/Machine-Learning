from sklearn.linear_model import Ridge, Lasso
import numpy as np

class My_Linear_Regression:
	def __init__(self, X_training, X_test, z, lambda_):
		self.X_training = X_training
		self.X_test = X_test
		self.z = z
		self.lambda_ = lambda_	

		"""

		If the data is split into training and test data 
		then this regression class will fit the model to 
		the training data and test the model on the test 
		data.
		If it is not plit, then X_test is sent in to the 
		class as the X_training.

		###### Variables #######

		X_training = training data used to fit the model
		X_test = data to test the model on
		z = the response
		lambda = the penalty
		"""

	def My_OLS(self):

		"""
		Ordinary Least Square method

		######  Method   #######
	
		Solves the OLS 
		beta = (X^TX)^-1X^Ty
	
		"""

		# Calculate the Ordinary Least Square             
		self.beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z)

	def My_Ridge(self):

		"""
		Ridge regression method

		######  Method   #######
	
		Solves the ridge regression
		beta = (X^TX + lambdaI)^-1X^Ty
		"""
		n = np.size(self.X_training,0)   # size of column (number of rows)
		m = np.size(self.X_training,1)   # number of columns
		# Separate the first column in X and first value in z
		X_ridge = self.X_training[:,1:]

		# Calculate mean of z_training equals beta_0
		beta_0 = 1.0/(n)*sum(self.z)
		I = np.identity(m-1)
		I_lambda = I*self.lambda_ 
		
		# Calculate the Ridge regression
		self.beta = np.linalg.inv(X_ridge.T.dot(X_ridge) + I_lambda).dot(X_ridge.T).dot(self.z)
		self.beta = np.insert(self.beta, 0, beta_0)

	def My_Lasso(self):

		"""
		Lasso regression method

		Calculate the Lasso regression using scikit learn
		"""
		
		lasso = Lasso(self.lambda_)
		self.beta = lasso.fit(self.X_training, self.z)

	def My_Predict(self, X_test, l):

		if l == True:
			z_predict = self.beta.predict(X_test)
	
		else:
			z_predict = X_test.dot(self.beta)

		return z_predict

		

             

