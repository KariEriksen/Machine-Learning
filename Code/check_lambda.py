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

lambda_array = np.linspace(lambda_, 1.0, 15)

L = len(lambda_array)
MSE = np.zeros(L)
bias = np.zeros(L)
variance = np.zeros(L)

for i in range(L):
	l = lambda_array[i]
	# Do bootstrap 
	boot = Bootstrap(X_fit, z, B, l, split, method)
	m, z_test = boot.My_Bootstrap()

	# Calculate different statistical properties
	MSE[i] = np.mean(np.mean((z_test - m)**2, axis=1, keepdims=True) )
	bias[i] = np.mean((z_test - np.mean(m, axis=1, keepdims=True))**2 )
	variance[i] = np.mean(np.var(m, axis=1, keepdims=True) )

print (MSE)
print (bias)
print (variance)

plt.plot(lambda_array, MSE, 'r--', lambda_array, bias, 'b--', lambda_array, variance, 'g--')
plt.show()









