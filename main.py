# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 20:05:10 2019

@author: Bhavdeep Singh
"""
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import random


NumpyFile = scipy.io.loadmat("AllSamples.mat")
#initial Plot
#turn on the interactive mode
plt.ion()
plt.scatter(NumpyFile['AllSamples'][:,0], NumpyFile['AllSamples'][:,1], alpha=0.5, s=100)
plt.title('Scatter plot for the given dataset')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

minRangeX = min(NumpyFile['AllSamples'][:,0])
maxRangeX = max(NumpyFile['AllSamples'][:,0])
minRangeY = min(NumpyFile['AllSamples'][:,1])
maxRangeY = max(NumpyFile['AllSamples'][:,1])

#source: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', \
          '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def computeObjectiveFunc(data, centroids):
    sum = 0
    for centroidNumber in range(0,len(centroids)):
        for value in data[centroidNumber]:
            sum += np.square(value[0]-centroids[centroidNumber][0])\
            + np.square(value[1]-centroids[centroidNumber][1])
    return sum

objectiveFunctionValue = []
# the number k of clusters ranging from 2-10
for k in range(2,11):
    #initialize the centroid value with random values in the given range of values
#    centroids = (max(maxRangeX,maxRangeY)-min(minRangeX,minRangeY))*\
#    np.random.rand(k,2)+min(minRangeX,minRangeY)
        # plot the position o
#    plt.scatter(NumpyFile['AllSamples'][:,0], NumpyFile['AllSamples'][:,1], color = (0,0.8,0.1), alpha=0.5)
#    plt.scatter(centroids[:,0],centroids[:,1])
#    plt.show()
    
#    centroids = np.ndarray.tolist(centroids)
    centroids = random.sample(list(NumpyFile['AllSamples']),k=k)

    while True:
        #this will store the list of values classified by the keyTh centroid
        classifications = {}
        
        #initialise the keys 
        for i in range(0,k):
            classifications[i] = []
            
        # cmpute nearest centroids
        for value in NumpyFile['AllSamples']:
            # credits to 
            # https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
            # for following line
            distancesFromCentroids = [np.linalg.norm(value-centroid) for centroid in centroids]
            # index will identify which centroid value it is assigned to
            nearestCentroid = distancesFromCentroids.index(min(distancesFromCentroids))
            classifications[nearestCentroid].append(value)
        
        newCentroids = []
        for i in range(0,k):
            if len(classifications[i])!=0:
                newCentroids.append(list(np.mean(classifications[i], axis=0)))
        converged = False
        for newCentroid,centroid in zip(newCentroids,centroids):
#            print(centroid)
#            print(newCentroid)
            if (newCentroid[0]-centroid[0] and newCentroid[1]-centroid[1])!=0:
                converged=False
                centroids = newCentroids[:]
                break
            else: 
                converged=True
        if converged == True:
            break
    for classification in classifications:
        color = colors[classification]
        for points in classifications[classification]:
            plt.scatter(points[0],points[1],color=color,s=200)
    for centroid in centroids:
        plt.scatter(centroid[0],centroid[1],marker='x',color='k',s=200)
    plt.title(k)
    plt.show()
#    print(classifications)
#    print(computeObjectiveFunc(classifications,centroids))
    objectiveFunctionValue.append(computeObjectiveFunc(classifications,centroids))


plt.plot(range(2,len(objectiveFunctionValue)+2),objectiveFunctionValue)
plt.scatter(range(2,len(objectiveFunctionValue)+2),objectiveFunctionValue,color='k')
plt.show()
    
    
    
    