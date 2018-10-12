from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.mlab as mlab
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

# Do bootstrap 
boot = Bootstrap(X_fit, z, B, lambda_, split, method)
m, z_test = boot.My_Bootstrap()

#print (z_test)

z_mat = np.tile(z_test, (B, 1))

# Calculate different statistical properties
MSE = np.mean(np.mean((z_test - m)**2, axis=1, keepdims=True) )
bias = np.mean((z_test - np.mean(m, axis=1, keepdims=True))**2 )
variance = np.mean(np.var(m, axis=1, keepdims=True) )
doubleR = 1.0 - sum((sum((z_mat - m)**2))/sum(sum((z_mat - np.mean(m, axis=1, keepdims=True))**2)))
#R2 = np.mean(doubleR)
"""
a = (z_test - m)**2
b = (z_test - np.mean(z_test))**2
sum1 = a.sum(axis=1)
sum2 = sum1.sum()
sum3 = b.sum()
doubleR = 1.0 - (float(sum2)/float(sum3))
"""
print ('Statistical properties')
print ('                      ') 
print ('Bias = %s' % bias)
print ('Variance = %s' % variance) 
print ('MSE = %s' % MSE) 
print ('Bias + Variance = %s' % (bias + variance)) 
print ('R2 = %s' % doubleR) 


# the histogram of the bootstrapped  data 
#plt.hist(np.mean(m, axis=1), normed=1, facecolor='red')
mean_m = np.mean(m, axis=1)
n, binsboot, patches = plt.hist(mean_m, 50, normed=1, facecolor='red', alpha=0.75)
# add a 'best fit' line  

t = mlab.normpdf( binsboot, np.mean(mean_m), np.std(mean_m))  
lt = plt.plot(binsboot, t, 'r--', linewidth=1)
plt.title('Mean of z_predict using Bootstrap, 100 iterations')
plt.xlabel('Mean predicted z')
plt.ylabel('Probability')
                                                                                                             
#plt.axis([99.5, 100.6, 0, 3.0])
plt.grid(True)
#plt.show()




