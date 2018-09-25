from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
import sys


fig = plt.figure()

# Read command line arguments ('method', number of x and y values, degree of polynom, alpha parameter)

method = sys.argv[1]
n = int(sys.argv[2])
d = int(sys.argv[3])
alpha = float(sys.argv[4])

# Produce data
#x = np.random.rand(n)                                             
#y = np.random.rand(n) 
step = 1.0/n
x = np.arange(0, 1, step) 
y = np.arange(0, 1, step)
x, y = np.meshgrid(x,y)  

x = np.reshape(x, np.size(x))
y = np.reshape(y, np.size(y)) 

X = np.c_[np.ones((n*n,1)), x, y, \
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
                    
beta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(z)
z_predict = X.dot(beta)

x_new = np.reshape(x, (n, n))
y_new = np.reshape(y, (n, n))
z_new = np.reshape(z, (n, n))

z_p_new = np.reshape(z_predict, (n, n))

# Plot the surface 
ax = fig.add_subplot(2, 1, 1, projection='3d')
surf = ax.plot_surface(x_new, y_new, z_new, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.plot_surface(x_new, y_new, z_p_new, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

plt.show()


