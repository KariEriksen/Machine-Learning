import numpy as np
#import scipy.sparse as sp
np.random.seed(12)
from linear_regression import My_Linear_Regression
from sklearn.linear_model import Lasso
from bootstrap import Bootstrap
import matplotlib.pyplot as plt
import sys

# decide which method to use
#part = 1 #OLS
#part = 2 #Ridge
#part = 3 #Lasso
#part = 4 #R score with bootstrap
#part = 5 #MSE with bootstrap

part = int(sys.argv[1])

# define Ising model aprams
# system size
L=40
lambda_ = 0.001

# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
	# only nearest neightbors have interaction
        J[i,(i+1)%L]-=1.0
	
    # compute energies, Einstein summation
    E = np.einsum('...i,ij,...j->...',states,J,states)

    return E

# calculate Ising energies
energies=ising_energies(states,L)

# reshape Ising states into RL samples: S_iS_j --> X_p
states=np.einsum('...i,...j->...ij', states, states)
shape=states.shape
states=states.reshape((shape[0],shape[1]*shape[2]))

# build final data set
Data=[states,energies]
coefs_leastsq = []
coefs_ridge = []
coefs_lasso = []

# define number of samples
n_samples=400

# define train and test data sets
X_train=Data[0][:n_samples]
Y_train=Data[1][:n_samples]              #+ np.random.normal(0,4.0,size=X_train.shape[0])
X_test=Data[0][n_samples:3*n_samples//2]
Y_test=Data[1][n_samples:3*n_samples//2] #+ np.random.normal(0,4.0,size=X_test.shape[0])

if part == 1:

	"""
	###### Part 1 #######

	Uses the OLS of the training data and plots 
	the J-matrix fitted to the model. 
	"""

	lr = My_Linear_Regression(X_train, X_test, Y_train, lambda_)
	lr.My_OLS()
	energies_predict = lr.My_Predict(X_test, False)
	coeff = lr.My_Beta()
	J_new=np.array(coeff).reshape((L,L))
	# plot 
	fig = plt.figure()
	cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
	plt.imshow(J_new,**cmap_args)
	plt.title('$\\mathrm{OLS}$', fontsize=12)
	plt.colorbar()
	plt.show()

elif part == 2:

	"""
	###### Part 2 #######

	Uses the ridge regression of the training data and plots 
	the J-matrix fitted to the model for different lambdas. 
	"""

	lambda_ = np.array([0.001, 0.01, 0.1, 1.0])
	coeff = np.zeros((4, np.size(X_test,1)))
	for i in range(4):
		lmbd = lambda_[i]
		lr = My_Linear_Regression(X_train, X_test, Y_train, lmbd)
		lr.My_Ridge()
		energies_predict = lr.My_Predict(X_test, False)	
		coeff[i,:] = lr.My_Beta()
	J_001 = np.array(coeff[0,:]).reshape((L,L))
	J_01 = np.array(coeff[1,:]).reshape((L,L))
	J_1 = np.array(coeff[2,:]).reshape((L,L))
	J_10 = np.array(coeff[3,:]).reshape((L,L))

	# plot
	cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
	plt.subplot(221)
	plt.imshow(J_001,**cmap_args)
	plt.title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lambda_[0]),fontsize=12)
	plt.tick_params(labelsize=10)
	
	plt.subplot(222)
	plt.imshow(J_01,**cmap_args)
	plt.title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lambda_[1]),fontsize=12)
	plt.tick_params(labelsize=10)
	plt.colorbar()

	plt.subplot(223)
	plt.imshow(J_1,**cmap_args)
	plt.title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lambda_[2]),fontsize=12)
	plt.tick_params(labelsize=10)

	plt.subplot(224)
	plt.imshow(J_10,**cmap_args)
	plt.title('$\\mathrm{Ridge},\ \\lambda=%.4f$' %(lambda_[3]),fontsize=12)
	plt.tick_params(labelsize=10)

	plt.subplots_adjust(left=0.05, right=0.85, wspace=0.05, hspace=0.5)
	plt.colorbar()
	plt.show()

elif part == 3:

	"""
	###### Part 3 #######

	Uses the lasso regression of the training data and plots 
	the J-matrix fitted to the model for different lambdas.
	"""

	lambda_ = np.array([0.001, 0.01, 0.1, 1.0])
	coeff = np.zeros((4, np.size(X_test,1)))
	doubleR = np.zeros(4)
	for i in range(4):
		lmbd = lambda_[i]
		lr = My_Linear_Regression(X_train, X_test, Y_train, lmbd)
		lr.My_Lasso()
		energies_predict = lr.My_Predict(X_test, True)
		coeff[i,:] = lr.My_Beta()
		doubleR[i] = 1.0 - ((Y_test - energies_predict)**2).sum()/((Y_test - Y_test.mean())**2).sum()
	#print ('lasso my score = %s' % doubleR)
	J_001 = np.array(coeff[0,:]).reshape((L,L))
	J_01 = np.array(coeff[1,:]).reshape((L,L))
	J_1 = np.array(coeff[2,:]).reshape((L,L))
	J_10 = np.array(coeff[3,:]).reshape((L,L))

	# plot
	cmap_args=dict(vmin=-1., vmax=1., cmap='seismic')
	plt.subplot(221)
	plt.imshow(J_001,**cmap_args)
	plt.title('$\\mathrm{Lasso},\ \\lambda=%.4f$' %(lambda_[0]),fontsize=12)
	plt.tick_params(labelsize=10)
	
	plt.subplot(222)
	plt.imshow(J_01,**cmap_args)
	plt.title('$\\mathrm{Lasso},\ \\lambda=%.4f$' %(lambda_[1]),fontsize=12)
	plt.tick_params(labelsize=10)
	plt.colorbar()

	plt.subplot(223)
	plt.imshow(J_1,**cmap_args)
	plt.title('$\\mathrm{Lasso},\ \\lambda=%.4f$' %(lambda_[2]),fontsize=12)
	plt.tick_params(labelsize=10)

	plt.subplot(224)
	plt.imshow(J_10,**cmap_args)
	plt.title('$\\mathrm{Lasso},\ \\lambda=%.4f$' %(lambda_[3]),fontsize=12)
	plt.tick_params(labelsize=10)

	plt.subplots_adjust(left=0.05, right=0.85, wspace=0.05, hspace=0.5)
	plt.colorbar()
	plt.show()

elif part == 4:

	"""
	###### Part 4 #######

	Uses the OLS, ridge and lasso regression of the training data and plots 
	the R-score for test data and predicted data, for different lambdas.
	"""

	method = 'OLS' 
	doubleR_OLS = np.zeros(9)
	lambda_ = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
	Y_mat = np.tile(Y_test, (100, 1))
	for i in range(9):
		# do bootstrap 
		boot = Bootstrap(X_train, X_test, Y_train, 100, lambda_[i], method)
		Y_pred, coeff_train = boot.My_Bootstrap()
		doubleR_OLS[i] = 1.0 - ((Y_mat - Y_pred)**2).sum()/((Y_mat - Y_mat.mean())**2).sum()

	method = 'Ridge' 
	doubleR_ridge = np.zeros(9)
	for i in range(9):
		# do bootstrap 
		boot = Bootstrap(X_train, X_test, Y_train, 100, lambda_[i], method)
		Y_pred, coeff_train = boot.My_Bootstrap()
		doubleR_ridge[i] = 1.0 - ((Y_mat - Y_pred)**2).sum()/((Y_mat - Y_mat.mean())**2).sum()

	method = 'Lasso' 
	doubleR_lasso = np.zeros(9)
	for i in range(9):
		# do bootstrap 
		boot = Bootstrap(X_train, X_test, Y_train, 100, lambda_[i], method)
		Y_pred, coeff_train = boot.My_Bootstrap()
		doubleR_lasso[i] = 1.0 - ((Y_mat - Y_pred)**2).sum()/((Y_mat - Y_mat.mean())**2).sum()

	# plot
	plt.semilogx(lambda_, doubleR_OLS, 'm--', label='OLS')
	plt.semilogx(lambda_, doubleR_ridge, 'c--', label='Ridge')
	plt.semilogx(lambda_, doubleR_lasso, 'g--', label='Lasso')
	plt.title('$R^2-score$')
	plt.xlabel('$\\lambda$')
	plt.ylabel('Error $\mathbf{y} - \mathbf{\hat{y}}$')
	plt.legend(loc='lower left')
	plt.show()

elif part == 5:

	"""
	###### Part 5 #######

	Run bootstrap one all methods, OLS, ridge and lasso, calculates 	
	the MSE, bias and variance of the coefficients from the fit 
	for different lambdas.
	"""

	method = 'Lasso' # select the optimal method
	MSE = np.zeros(9)
	bias = np.zeros(9)
	variance = np.zeros(9)
	lambda_ = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
	for i in range(9):
		# do bootstrap 
		boot = Bootstrap(X_train, X_test, Y_train, 100, lambda_[i], method)
		Y_pred, coeff = boot.My_Bootstrap()
		#Y_mat = np.tile(Y_test, (100, 1))
		# Calculate different statistical properties
		MSE[i] = np.mean(np.mean((-1 - coeff)**2, axis=1, keepdims=True) )
		bias[i] = np.mean((-1 - np.mean(coeff, axis=1, keepdims=True))**2 )
		variance[i] = np.mean(np.var(coeff, axis=1, keepdims=True) )

	# plot
	plt.semilogx(lambda_, MSE, 'm--', label='MSE')
	plt.semilogx(lambda_, bias, 'c--', label='$bias^2$')
	plt.semilogx(lambda_, variance, 'g--', label='variance')
	plt.title('Bias-variance tradeoff, ridge')
	plt.xlabel('$\\lambda$')
	plt.ylabel('Error in coefficients')
	plt.legend(loc='center left')
	plt.show()

else:
	print('Part of project must be given, ex. 1, corresponds to OLS')








