# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 18:34:53 2019

@author: Bhavdeep Singh
"""

#importing the libraries 
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from future.utils import iteritems
import math

class NaiveBayes():
    def __calculateLogProbabilityDensity(self, x, mean, stdev):
        if stdev == 0:
            return 0
        else:
            exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
            if (1 / (math.sqrt(2*math.pi) * stdev)) * exponent == 0:
                return 0
            else:
                return math.log((1 / (math.sqrt(2*math.pi) * stdev)) * exponent)
            
    def fit(self, X, Y):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for label in labels:
            current_x = X[Y == label]
            self.gaussians[label] = {
                'mean': np.mean(current_x,axis=0),
                'std': np.std(current_x,axis=0),
            }
            self.priors[label] = float(len(Y[Y == label])) / len(Y)
        
#        print(self.gaussians)
#        print(len(self.gaussians[0]['mean']))
#        print(self.priors)
    
    def predict(self, X):
        prediction = []
        for sample in X:
            probabilities = {}
            for label in self.gaussians.keys():
                listMeanPerLabel = self.gaussians[label]['mean'];
                listStdPerLabel = self.gaussians[label]['std'];
                probabilities[label] = 0
                for pixel,mean,std in zip(sample,listMeanPerLabel,listStdPerLabel):
                    probabilities[label] = probabilities[label] + self.__calculateLogProbabilityDensity(pixel,mean,std)
                probabilities[label] = probabilities[label] + self.priors[label]
            prediction.append(max(probabilities, key=probabilities.get))
        return prediction

class NaiveBayes2():
    #Function to apply 1-d Gaussian Normal Distribution
    def multivariate_normal(self,x,mean,covariance):
       x_m=np.subtract(x,mean)
       return (1/(np.sqrt((2*np.pi)**2 * np.linalg.det(covariance)))*np.exp(-((x_m).T.dot(np.linalg.inv(covariance)).dot(x_m))/2))

    def fit(self, X, Y):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for label in labels:
            current_x = X[Y == label]
            self.gaussians[label] = {
                'mean': np.mean(current_x,axis=1),
                'std': np.std(current_x,axis=1),
            }
            self.priors[label] = float(len(Y[Y == label])) / len(Y)           
#        print(self.gaussians)
#        print(len(self.gaussians[0]['mean']))
#        print(self.priors)
        self.multivariateMean()
        
    def multivariateMean(self):
        self.multivariate = {}
        self.covariance = {}
        for label in self.gaussians.keys():
            self.multivariate[label] = [np.mean(self.gaussians[label]['mean']),np.mean(self.gaussians[label]['std'])]
            print(self.gaussians[label]['mean'])
            self.covariance[label] = np.cov(self.gaussians[label]['mean'],self.gaussians[label]['std'])
#            print(self.covariance[label])
            for i in range(len(self.covariance[label])):
                for j in range(len(self.covariance[label])):
                    if i!=j:
                        self.covariance[label][i][j] = 0
                        
    def predict(self,X):
        prediction = []
        for sample in X:
            testSample = [np.mean(sample),np.std(sample)]
            probabilities = {}
            for label in self.multivariate.keys():
                probabilities[label] = self.multivariate_normal(testSample, self.multivariate[label], self.covariance[label])
            prediction.append(max(probabilities, key=probabilities.get))
        return prediction
    
class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
            
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            loss = self.__loss(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X).round()

def extractFeatures(data):
    extractedData = data
    listMean = np.mean(extractedData,axis = 1)
    listStd = np.std(extractedData,axis = 1)
    
    return np.column_stack((listMean,listStd))



def confusionMatrix(results, labelVector):
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0
    for value, label in zip(results,labelVector):#NumpyFile['tsY'][0]
        if value == label:
            if value == 0:
                truePositive = truePositive + 1
            else:
                trueNegative = trueNegative + 1
        else:
            if label == 0:
                falsePositive = falsePositive + 1
            else:
                falseNegative = falseNegative + 1
    
    print("Overall Accuracy : "+str((truePositive+trueNegative)/(truePositive+\
                                    trueNegative+falseNegative+falsePositive)))
    
    print("Counfusion Matrix : ")
    cnfMatrx = np.array([[truePositive,falsePositive],[falseNegative,trueNegative]])
    print(cnfMatrx)
    
if __name__=="__main__":
    
    """
        mnist_data.mat Description:
            
            trX - training set, each row represents a digit
            trY - training labels, 0 and 1 represent digit 7 and 8 respectively
            tsX - testing set, each row represents a digit
            tsY - testing labels, 0 and 1 represent digit 7 and 8 respectively
            
            each row is a different image.         
    """
    ## loding the data with the above mentioned characteristics
    NumpyFile = scipy.io.loadmat("mnist_data.mat")
    
    
    
    ## this region is to demostrate what we are learning ghraphically
    data7 = []
    data8 = []
    for image, label in zip(NumpyFile['trX'],NumpyFile['trY'][0]):
        if label == 0:
            data7.append(image)
        else:
            data8.append(image)
    
    print("Total number of images with 7 are : " + str(len(data7)))
    print("Total number of images with 8 are : " + str(len(data8)))
    print("Total number of images : " + str(len(NumpyFile['trX'])))

    meanVector = np.mean(NumpyFile['trX'], axis=1)
    stdVector = np.std(NumpyFile['trX'], axis=1)
    
    meanOfPixelsFor7 = np.mean(data7,axis=0)
    stdOfPixelsFor7 = np.std(data7,axis=0)
    meanOfPixelsFor8 = np.mean(data8,axis=0)
    stdOfPixelsFor8 = np.std(data8,axis=0)
    
    # if we take pixelwise mean this is what the model learns for the 8
    plt.imshow(meanOfPixelsFor8.reshape(28,28))
    plt.title("8")
    plt.show()
    
    # if we take pixelwise mean this is what the model learns for the 7
    plt.imshow(meanOfPixelsFor7.reshape(28,28))
    plt.title("7")
    plt.show()

    ## this is not the actual solution but this is naive bayes with every pixel
    ## taken as a feature
    model = NaiveBayes()
    model.fit(NumpyFile['trX'],NumpyFile['trY'][0])
    results = model.predict(NumpyFile['tsX'])
    
    confusionMatrix(results,NumpyFile['tsY'][0])
    
    ## feature extraction of the training data
    
    features = extractFeatures(NumpyFile['trX'])
    
    ## This is the Naive Bayes solution
    model = NaiveBayes2()
    model.fit(NumpyFile['trX'],NumpyFile['trY'][0])
    results = model.predict(NumpyFile['tsX'])
    
    confusionMatrix(results,NumpyFile['tsY'][0])
        
    ## This is the logistic regression solution
    model = LogisticRegression(lr=0.1, num_iter=300000)
    
    model.fit( features, NumpyFile['trY'][0])
    
    features = extractFeatures(NumpyFile['tsX'])
#    print(features)
    
    results = model.predict(features)
    confusionMatrix(results,NumpyFile['tsY'][0])
#    print(results)
