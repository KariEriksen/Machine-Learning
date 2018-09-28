from sklearn.linear_model import Ridge, Lasso
import numpy as np

class My_Linear_Regression:
	def __init__(self, X_training, X_test, z, alpha):
		self.X_training = X_training
		self.X_test = X_test
		self.z = z
		self.alpha = alpha	

		"""
		Check if the data sample is split in to training
		and test data. If not, then the two variables are 		the same and we do at fit and predict on the same
		data.
		"""
		if X_test == None:
			self.X_test = X_training	

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
		beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z)
		z_predict = self.X_test.dot(beta)
		
		return z_predict

	def My_Ridge(self):

		"""
		Ridge regression method
		OBS!!!!! Add lambda
		"""

		# Calculate the Ridge regression
		beta = np.linalg.inv(self.X_training.T.dot(self.X_training)).dot(self.X_training.T).dot(self.z)
		z_predict = self.X_test.dot(beta)
		
		return z_predict

	def My_Lasso(self):

		# Calculate the Lasso regression using scikit learn
		lasso = Lasso(alpha)
		lasso.fit(X_training, z_n)
		z_predict = lasso.predict(X_test)

		return z_predict

             

