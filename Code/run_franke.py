from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import sys

from linear_regression import My_Linear_Regression 
from bootstrap import Bootstrap 

# Read command line arguments ('method', number of x and y values, alpha parameter)

# Take in command line arguments
method = sys.argv[1]
n = int(sys.argv[2])
lambda_ = float(sys.argv[3])

# Set some values
B = 100
split = 0.7

# Produce data
step = 1.0/n
x = np.arange(0, 1, step)             
y = np.arange(0, 1, step)                                             

x, y = np.meshgrid(x,y)   

x = np.reshape(x, np.size(x))
y = np.reshape(y, np.size(y)) 

# Fit the design matrix
X_fit = np.c_[np.ones((n*n,1)), x, y, \
		      x**2, x*y, y**2, \
		      x**3, x**2*y, x*y**2, y**3, \
		      x**4, x**3*y, x**2*y**2, x*y**3, y**4, \
		      x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5]                                                     

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

#noise = np.asarray(random.sample((range(n)),n))
noise = np.random.random_sample((n,))
z = FrankeFunction(x, y) 

# Do linear regression
"""
lr = My_Linear_Regression(X_fit, None, z, lambda_)

if method == 'OLS':
	lr.My_OLS()
	z_predict = lr.My_Predict(X_fit, False)

elif method == 'Ridge':
	lr.My_Ridge()
	z_predict = lr.My_Predict(X_fit, False)	

elif method == 'Lasso':
	lr.My_Lasso()
	z_predict = lr.My_predict(X_fit, True)
	
diff = z - z_predict
MSE = 1.0/(n*n)*(sum(diff*diff))
print (MSE)
"""
#print ('z = %s' % z)
#print ('      ')
#print ('z_predict = %s' % z_predict)

# Do bootstrap 
boot = Bootstrap(X_fit, z, B, lambda_, split, method)
m, z_test = boot.My_Bootstrap()

# Calculate different statistical properties
MSE = np.mean(np.mean((z_test - m)**2, axis=1, keepdims=True) )
bias = np.mean((z_test - np.mean(m, axis=1, keepdims=True))**2 )
variance = np.mean(np.var(m, axis=1, keepdims=True) )
#R2 = np.mean(doubleR)
a = (z_test - m)**2
b = (z_test - np.mean(z_test))**2
sum1 = a.sum(axis=1)
sum2 = sum1.sum(axis=0)
sum3 = b.sum()
doubleR = 1.0 - sum2/sum3

print ('Statistical properties')
print ('                      ') 
print ('Bias = %s' % bias)
print ('Variance = %s' % variance) 
print ('MSE = %s' % MSE) 
print ('Bias + Variance = %s' % (bias + variance)) 
print ('R2 = %s' % doubleR) 





