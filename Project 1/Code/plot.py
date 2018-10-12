from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

from linear_regression import My_Linear_Regression 
from bootstrap import Bootstrap 
from design_matrix import Design_Matrix

fig = plt.figure()
#ax = fig.gca(projection='3d')

# Read command line arguments ('method', number of x and y values, alpha parameter)

# Take in command line arguments
method = sys.argv[1]
n = int(sys.argv[2])
lambda_ = float(sys.argv[3])
poly_degree = int(sys.argv[4])

# Produce data
step = 1.0/n
x = np.arange(0, 1, step)             
y = np.arange(0, 1, step)                                             

x, y = np.meshgrid(x,y)   

x = np.reshape(x, np.size(x))
y = np.reshape(y, np.size(y)) 

d = poly_degree

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
lr = My_Linear_Regression(X_fit, None, z, lambda_)

if method == 'OLS':
	lr.My_OLS()
	z_predict = lr.My_Predict(X_fit, False)

elif method == 'Ridge':
	lr.My_Ridge()
	z_predict = lr.My_Predict(X_fit, False)	

elif method == 'Lasso':
	lr.My_Lasso()
	z_predict = lr.My_Predict(X_fit, True)


MSE = np.mean((z - z_predict)**2)
bias = np.mean((z - np.mean(z_predict))**2)
variance = np.mean(np.var(z_predict))
doubleR = 1.0 - sum((z - z_predict)**2)/sum((z_predict - np.mean(z_predict))**2)

print ('Statistical properties')
print ('                      ') 
print ('Bias = %s' % bias)
print ('Variance = %s' % variance) 
print ('MSE = %s' % MSE) 
print ('Bias + Variance = %s' % (bias + variance)) 
print ('R2 = %s' % doubleR) 

x = np.reshape(x, (n, n))
y = np.reshape(y, (n, n))
z = np.reshape(z, (n, n))
z_new = np.reshape(z_predict, (n, n))

#1 subplot, plot the surface
ax = fig.add_subplot(2, 1, 1, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

# Add title and labels
ax.text2D(0.25, 0.95, "Surface plot of Franke function", transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')

#2 subplot, plot the regression fit
ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.plot_surface(x, y, z_new, cmap=cm.coolwarm, linewidth=0, antialiased=False)

#ax.set_zlim(-0.10, 1.40)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add title and labels
ax.text2D(0.1, 0.95, "Surface plot of the regression fit using %s up to %s'th order" % (method, d), transform=ax.transAxes)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('ẑ')

plt.show()




