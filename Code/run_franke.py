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
from sklearn.metrics import mean_squared_error
import sys

from My_Linear_Regression import My_OLS, My_Ridge, My_Lasso

# Read command line arguments ('method', number of x and y values, alpha parameter)

# Take in command line arguments
method = sys.argv[1]
n = int(sys.argv[2])
alpha = float(sys.argv[4])

# Produce data
step = 1.0/n
x = np.arange(0, 1, step)             
y = np.arange(0, 1, step)                                             

x, y = np.meshgrid(x,y)                                               

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
lr = My_Linear_Regression()
MSE = lr.My_('method') #something something






