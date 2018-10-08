from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import sys

from linear_regression import My_Linear_Regression 
from bootstrap import Bootstrap 
from design_matrix import Design_Matrix

# Read command line arguments ('method', number of x and y values, alpha parameter)

# Take in command line arguments
method = sys.argv[1]
n = int(sys.argv[2])
lambda_ = float(sys.argv[3])
poly_degree = int(sys.argv[4])

# Set some values
B = 100
split = 0.7

# Produce data
step = 1.0/n
x = np.arange(0, 1, step)             
y = np.arange(0, 1, step)                                             

x, y = np.meshgrid(x,y)

d = poly_degree   

x = np.reshape(x, np.size(x))
y = np.reshape(y, np.size(y)) 

# Fit the design matrix
matrix = Design_Matrix(x, y, d, n)
X_fit = matrix.fit_matrix()

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

noise = np.random.normal(0, 1, n*n)
z = FrankeFunction(x, y) + noise

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





