from sklearn.linear_model import Ridge, Lasso
import numpy as np

class My_Linear_Regression:
	def __init__(self, X_training, X_test, z_training, lambd):
		self.X_training = X_training
		self.X_test = X_test
		self.z_training = z_training
		self.lambd = lambd	

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
		self.beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z_training)


	def My_Ridge(self):

		"""
		Ridge regression method
		OBS!!!!! Add lambda
		"""
		# Remove first column

		# Calculate mean of z_training equals beat_0
		"""
		n = np.size(self.X_training,0)
		I = np.identity(n-1)
		I_lambda = I*self.lambd 
		X_ridge = X_new + I_lambda
		self.beta = np.linalg.inv(self.X_ridge.T.dot(self.X_ridge)).dot(self.X_ridge.T).dot(self.z_training)
		"""
		# Calculate the Ridge regression
		self.beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z_training)


	def My_Lasso(self):

		# Calculate the Lasso regression using scikit learn
		#lasso = Lasso(alpha)
		#beta = lasso.fit(X_training, z_n)
		#z_predict = lasso.predict(X_test)
		self.beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z_training)


	def My_Predict(self, X_test):
		z_predict = X_test.dot(self.beta)

		return z_predict

		

             

