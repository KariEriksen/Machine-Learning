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
step = 1.0/n
x = np.arange(0, 1, step)                                             # array (20,)
y = np.arange(0, 1, step)                                             # array (20,)

x, y = np.meshgrid(x,y)                                               # array (20, 20), (20, 20)
def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

#noise = np.asarray(random.sample((range(n)),n))
noise = np.random.random_sample((n,))
z = FrankeFunction(x, y)                                             # array (20, 20)

x_n = np.reshape(x, np.size(x))                                      # array (20*20, )
y_n = np.reshape(y, np.size(y))
z_n = np.reshape(z, np.size(z))

# Polynomial fit
poly = PolynomialFeatures(degree=d)
N = np.size(x)
X = np.c_[np.ones((N,1)),x_n,y_n]
X_fit = poly.fit_transform(X)

# Ordinary Least Square method
if method == 'OLS':
	OLS = LinearRegression()
	OLS.fit(X_fit,z_n)
	z_predict = OLS.predict(X_fit)

# Ridge regression
elif method == 'Ridge':
	ridge = Ridge(alpha)
	ridge.fit(X_fit, z_n)
	z_predict = ridge.predict(X_fit)

#Lasso regression
elif method == 'Lasso':

	lasso = Lasso(alpha)
	lasso.fit(X_fit, z_n)
	z_predict = lasso.predict(X_fit)

else:
	print('You have forgotten to select method; OLS, Ridge or Lasso.')

z_new = np.reshape(z_predict, (n, n))

# Plot the surface 
ax = fig.add_subplot(2, 1, 1, projection='3d')
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

ax = fig.add_subplot(2, 1, 2, projection='3d')
ax.plot_surface(x, y, z_new, cmap=cm.coolwarm,
linewidth=0, antialiased=False)

ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))


plt.show()



