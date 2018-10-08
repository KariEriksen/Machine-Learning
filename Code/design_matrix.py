import numpy as np

class Design_Matrix:
	def __init__(self, X, Y, d, N):
		self.X = X
		self.Y = Y
		self.d = d	
		self.N = N

	def fit_matrix(self):   

		x = self.X
		y = self.Y
		n = self.N

		if self.d == 1:
			X_fit = np.c_[np.ones((n*n,1)), x, y, x*y]

		elif self.d == 2:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2]  
	
		elif self.d == 3:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3]    
	
		elif self.d == 4:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4] 
	
		elif self.d == 5:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]  
	
		elif self.d == 6:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5, \
			      x**6, x**5*y, x**4*y**2, x**3*y**3, x**2*y**4, x*y**5, y**6]     

		elif self.d == 7:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5, \
			      x**6, x**5*y, x**4*y**2, x**3*y**3, x**2*y**4, x*y**5, y**6, \
			      x**7, x**6*y, x**5*y**2, x**4*y**3, x**3*y**4, x**2*y**5, x*y**6, y**7]      

		elif self.d == 8:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5, \
          	                x**6, x**5*y, x**4*y**2, x**3*y**3, x**2*y**4, x*y**5, y**6, \
          	                x**7, x**6*y, x**5*y**2, x**4*y**3, x**3*y**4, x**2*y**5, x*y**6, y**7, \
          	                x**8, x**7*y, x**6*y**2, x**5*y**3, x**4*y**4, x**3*y**5, x**2*y**6, x*y**7, y**8] 

		elif self.d == 9:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5, \
          	                x**6, x**5*y, x**4*y**2, x**3*y**3, x**2*y**4, x*y**5, y**6, \
          	                x**7, x**6*y, x**5*y**2, x**4*y**3, x**3*y**4, x**2*y**5, x*y**6, y**7, \
          	                x**8, x**7*y, x**6*y**2, x**5*y**3, x**4*y**4, x**3*y**5, x**2*y**6, x*y**7, y**8, \
          	                x**9, x**8*y, x**7*y**2, x**6*y**3, x**5*y**4, x**4*y**5, x**3*y**6, x**2*y**7, x*y**8, y**9] 

		elif self.d == 10:
			X_fit = np.c_[np.ones((n*n,1)), x, y, \
			      x**2, x*y, y**2, \
			      x**3, x**2*y, x*y**2, y**3, \
			      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
			      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5, \
          	                x**6, x**5*y, x**4*y**2, x**3*y**3, x**2*y**4, x*y**5, y**6, \
          	                x**7, x**6*y, x**5*y**2, x**4*y**3, x**3*y**4, x**2*y**5, x*y**6, y**7, \
          	                x**8, x**7*y, x**6*y**2, x**5*y**3, x**4*y**4, x**3*y**5, x**2*y**6, x*y**7, y**8, \
          	                x**9, x**8*y, x**7*y**2, x**6*y**3, x**5*y**4, x**4*y**5, x**3*y**6, x**2*y**7, x*y**8, y**9, \
          	                x**10, x**9*y, x**8*y**2, x**7*y**3, x**6*y**4, x**5*y**5, x**4*y**6, x**3*y**7, x**2*y**8, x*y**9, y**10] 

		else:
			print('You must give a polynomial degree.')

		return X_fit


