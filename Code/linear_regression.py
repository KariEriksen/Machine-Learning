from sklearn.linear_model import Ridge, Lasso
import numpy as np

class My_Linear_Regression:
	def __init__(self, X, alpha):
		self.X = X
		self.alpha = alpha	

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
		beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
		z_predict = X.dot(beta)
		
		return z_predict

	def My_Ridge(self):

		# Calculate the Ridge regression
		ridge = Ridge(alpha)
		ridge.fit(X_fit, z_n)
		z_predict = ridge.predict(X)

		return z_predict

	def My_Lasso(self):

		# Calculate the Lasso regression using scikit learn
		lasso = Lasso(alpha)
		lasso.fit(X_fit, z_n)
		z_predict = lasso.predict(X)

		return z_predict

             

