import numpy as np

class My_Linear_Regression:
	def __init__(self, X, alpha):
		self.X = X
		self.alpha = alpha	

	def My_OLS(self, X, alpha):

		"""
		Ordinary Least Square method
	 	
		###### Variables #######

		method = the selected regression method to use
		alpha = lambda in Ridge or Lasso regression 
	
		######  Method   #######
	
		Solves the OLS ....
		up to fifth order
	
		"""
	
		# Ordinary Least Square method
		OLS = LinearRegression()
		OLS.fit(X_fit,z_n)
		z_predict = OLS.predict(X_fit)
		
		return z_predict

	def My_Ridge(self, X, alpha):

		ridge = Ridge(alpha)
		ridge.fit(X_fit, z_n)
		z_predict = ridge.predict(X_fit)

		return z_predict

	def My_Lasso(self, X, alpha):

		ridge = Ridge(alpha)
		ridge.fit(X_fit, z_n)
		z_predict = ridge.predict(X_fit)

		return z_predict

             

