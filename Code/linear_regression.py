import numpy as np

class My_Linear_Regression:
    def __init__(self, X, method, alpha):
	self.method = method
	self.alpha = alpha	

    def My_OLS(self, method, alpha):

	"""
	Ordinary Least Square method
 	
	###### Variables #######

	method = the selected regression method to use
	alpha = lambda in Ridge or Lasso regression 
	
	######  Method   #######
	
	Solves the OLS ....
	up to fifth order
	
	"""

             

