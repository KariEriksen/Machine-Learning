import numpy as np
from numpy.random import randint
from linear_regression import My_Linear_Regression 

class Bootstrap:
    def __init__(self, X, z, B, lambda_, split, method):
        self.X = X
        self.z = z
        self.B = B
        self.lambda_ = lambda_	
        self.split = split
        self.method = method
	    
    def My_Bootstrap(self):

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

        # Split the data according to the given decimal percentage
        # split = amount of data set to training
        n = np.size(self.X,0)   # size of column (number of rows)
        C = int(n*self.split)

        # Shuffle the datapoint to randomized order
        randomize = np.arange(n)
        np.random.shuffle(randomize)
        self.X = self.X[randomize]
        self.z = self.z[randomize]

        self.X_training = self.X[:C,:]
        self.X_test = self.X[C:,:]
        self.z_training = self.z[:C]
        self.z_test = self.z[C:]

        # Do linear regression
        t = len(self.z_test)
        # Ordinary Least Square method
        if self.method == 'OLS':
            m = np.zeros((self.B,t))	
            for i in range(self.B):
                index = randint(0, C, C)
                X_resample = self.X_training[index]
                z_resample = self.z_training[index]
                lr = My_Linear_Regression(X_resample, self.X_test, z_resample, self.lambda_)
                lr.My_OLS()
                z_predict = lr.My_Predict(self.X_test, False)
                m[i,:] = z_predict

            # Calculate different statistical properties
            mean_z =  1.0/t*sum(z_predict)
            bias =    1.0/t*sum((self.z_test - mean_z)**2)
            var =     1.0/t*sum((z_predict - mean_z)**2)
            MSE =     1.0/t*sum((self.z_test - z_predict)**2)
            doubleR = 1.0 - (sum((self.z_test - z_predict)**2)/sum((self.z_test - mean_z)**2)) 
        	
        # Ridge regression
        elif self.method == 'Ridge':
            z_predict = lr.My_Ridge()
            m = zeros(B)	
            for i in range(B):
                z_predict = lr.My_Predict(self.X_test)			
                t = self.My_Bootstrap_Method(z_predict)
                m[i] = mean(t)
        
        #Lasso regression
        elif self.method == 'Lasso':
            z_predict = lr.My_Lasso()
            m = zeros(B)	
            for i in range(B):
                z_predict = lr.My_Predict(self.X_test)
                t = self.My_Bootstrap_Method(z_predict)
                m[i] = mean(t)
        
        else:
            print('You have forgotten to select method; OLS, Ridge or Lasso.')
        # Calculate the mean and standard deviation
        #mean = 1.0/n*(sum(t))  
        #std = 1.0/n*(sum())
        
        # Calculate the Mean Square Error
        #diff = self.z - self.z_predict
        #diff = np.mean(self.z) - np.mean(m)
        #MSE = 1.0/(n*n)*(diff*diff)
        
        return m




