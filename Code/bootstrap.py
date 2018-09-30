import numpy as np
from numpy.random import randint
from linear_regression import My_Linear_Regression 

class Bootstrap:
    def __init__(self, X, z, B, alpha, split, method):
        self.X = X
        self.z = z
        self.B = B
        self.alpha = alpha
        self.split = split
        self.method = method
	    
    def My_Bootstrap_Method(self, z_predict):

        """
        Bootstrap class
	
        ###### Variables #######
        data is an array containing all data from experiment
        B = number of bootstraps
        split = amount split to training data in decimal percentage
        
        ######  Method   #######
	
        Resampling method with replacement, takes data sample
        returns the standard deviation and mean of the resampled
        Samples values with replacement and calculates the mean 
        Repeated n times
        Generates sampling distribution of mean
        """

        t = zeros(n)
        for i in range(n):
            sample = self.z_predict[randint(0,n,n)]
            t[i] = (1/n)*np.sum(sample)
        return t


    def My_Bootstrap(self):
        # Split the data according to the given decimal percentage
        # split = amount of data set to training
        n = np.size(self.X,0)   # size of column (number of rows)
        C = int(n*self.split)

        np.random.shuffle(self.X)
        self.X_training = self.X[:C,:]
        self.X_test = self.X[C:,:]
        
        # Do linear regression
        lr = My_Linear_Regression(self.X_training, self.X_test, self.z, self.alpha)

        # Ordinary Least Square method
        if self.method == 'OLS':
            lr.My_OLS()
            m = zeros(self.B)	
            for i in range(self.B):
                z_predict = lr.My_Predict(self.X_test)
                t = My_Bootstrap_Method(z_predict)
                m[i] = (1/self.B)*sum(t)
        	
        # Ridge regression
        elif self.method == 'Ridge':
            z_predict = lr.My_Ridge()
            m = zeros(B)	
            for i in range(B):
                z_predict = lr.My_Predict(self.X_test)			
                t = My_Bootstrap_Method(z_predict)
                m[i] = mean(t)
        
        #Lasso regression
        elif self.method == 'Lasso':
            z_predict = lr.My_Lasso()
            m = zeros(B)	
            for i in range(B):
                z_predict = lr.My_Predict(self.X_test)
                t = My_Bootstrap_Method(z_predict)
                m[i] = mean(t)
        
        else:
            print('You have forgotten to select method; OLS, Ridge or Lasso.')
        # Calculate the mean and standard deviation
        #mean = 1.0/n*(sum(t))  
        #std = 1.0/n*(sum())
        
        # Calculate the Mean Square Error
        #diff = self.z - self.z_predict
        diff = np.mean(z) - np.mean(m)
        MSE = 1/(n*n)*(sum(diff*diff))
        
        return MSE




