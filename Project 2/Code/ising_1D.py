import numpy as np
#import scipy.sparse as sp
np.random.seed(12)
from linear_regression import My_Linear_Regression
from sklearn.linear_model import Lasso
from bootstrap import Bootstrap
import matplotlib.pyplot as plt

# decide which method to use
#method = 1 # OLS
#method = 2 #Ridge
#method = 3 #Lasso
method = 4 #Bootstrap with Lasso

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

if method == 1:
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
	plt.show()

elif method == 2:
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

	plt.subplots_adjust(wspace=0.05, hspace=0.5)
	plt.colorbar()
	plt.show()

elif method == 3:
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

	plt.subplots_adjust(wspace=0.05, hspace=0.5)
	plt.colorbar()
	#plt.show()

elif method == 4:
	method = 'Lasso' # select the optimal method
	#MSE = np.zeros(10)
	#bias = np.zeros(10)
	#variance = np.zeros(10)
	#MSE_test = np.zeros(10)
	#bias_test = np.zeros(10)
	#variance_test = np.zeros(10)
	doubleR = np.zeros(9)
	lambda_ = np.array([0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
	for i in range(9):
		# do bootstrap 
		boot = Bootstrap(X_train, X_test, Y_train, 100, lambda_[i], method)
		Y_pred, coeff = boot.My_Bootstrap()

		# Calculate different statistical properties
		#MSE[i] = np.mean(np.mean((Y_test - Y_pred)**2, axis=1, keepdims=True) )
		#bias[i] = np.mean((Y_test - np.mean(Y_pred, axis=1, keepdims=True))**2 )
		#variance[i] = np.mean(np.var(Y_pred, axis=1, keepdims=True) )

		#MSE[i]_test = np.mean(np.mean((Y_test - Y_pred)**2, axis=1, keepdims=True) )
		#bias[i]_test = np.mean((Y_test - np.mean(Y_pred, axis=1, keepdims=True))**2 )
		#variance[i]_test = np.mean(np.var(Y_pred, axis=1, keepdims=True) )
		Y_mat = np.tile(Y_test, (100, 1))
		#doubleR[i] = 1.0 - sum((sum((Y_mat - Y_pred)**2))/sum(sum((Y_mat - np.mean(Y_mat, axis=1, keepdims=True))**2)))
		doubleR[i] = 1.0 - ((Y_mat - Y_pred)**2).sum()/((Y_mat - Y_mat.mean())**2).sum()
		"""
		MSE[i] = np.mean(np.mean((-1 - coeff)**2, axis=1, keepdims=True) )
		bias[i] = np.mean((-1 - np.mean(coeff, axis=1, keepdims=True))**2 )
		variance[i] = np.mean(np.var(coeff, axis=1, keepdims=True) )
		"""
	# plot
	"""
	plt.plot(lambda_, MSE, 'r--', lambda_, bias, 'b--', lambda_, variance, 'g--')
	plt.legend(('MSE', 'bias^2', 'variance'), loc='upper right')
	plt.title('Bias-variance tradeoff')
	plt.xlabel('$\\lambda$')
	plt.ylabel('Error')
	plt.show()
	"""
	plt.plot(lambda_, doubleR, 'r--',)
	plt.title('$R^2 score$')
	plt.xlabel('$\\lambda$')
	plt.ylabel('Error')
	plt.show()

else:
	print('Method must be given, ex. 1, corresponds to OLS')








