#import packages
import numpy as np
import matplotlib.pyplot as plt
import random

def dataprocessing():
    #This function Takes in data and normalizes it for easier distance calculation
    #parameters: None
    #return: fdata - the normalized data
    #It returns the normalized data
    f = open("data.txt", "r+", encoding ="utf-8")
    raw = f.read().split()
    f.close()
    data = np.array(raw)
    a1 =[]
    a2 =[]
    na1 = []
    na2 = []
    dataset = []
    #transform the data to float value
    for i in range (len(data)):
        k = data[i].split(",")
        a1.append(float(k[0]))
        a2.append(float(k[1]))
    #normalize the data
    for i in range (len(data)):
        na1.append(round((a1[i]-min(a1))/(max(a1) - min(a1)),3))
        na2.append(round((a2[i]-min(a2))/(max(a2) - min(a2)),3))
    #turn data into an array dataset
    for i in range (len(data)):
        d =[]
        d.append(na1[i])
        d.append(na2[i])
        dataset.append(d)
    fdata = np.array(dataset)
    #return the normalized data
    return fdata

def kmeans(data,k):
    #This function is a implementation of The K-means algorithm
    #Prameters: data - the data set and k - number of clusters
    #Return:
    # - centers: array of final centers
    # - count: number of iterations
    # - cluster:labels (the assignment of points to centers)
    # - errorvalL: final error value
    #I implement this code based on the textbook code
    errorval = 0
    # Find the minimum and maximum values for each feature
    minima = data.min(axis=0)
    maxima = data.max(axis=0)
    # Pick the centre locations randomly
    centers = np.random.rand(k,2)*(maxima-minima)+minima
    oldCenters = np.random.rand(k,2)*(maxima-minima)+minima
    nData = len(data)
    count = 0
    #loop until the centers stop moving
    while np.sum(np.sum(oldCenters-centers))!= 0:

        oldCenters = centers.copy()
        count += 1
            
        # Compute distances
        distances = np.ones((1,nData))*np.sum((data-centers[0,:])**2,axis=1)
        for j in range(k-1):
            distances = np.append(distances,np.ones((1,nData))*np.sum((data-centers[j+1,:])**2,axis=1),axis=0)
        # Identify the closest cluster
        cls = distances.argmin(axis=0)
        cluster = np.transpose(cls*np.ones((1,nData)))
        
        # Update the cluster centres
        for j in range(k):
            thisCluster = np.where(cluster==j,1,0)
            if sum(thisCluster)>0:
                centers[j,:] = np.sum(data*thisCluster,axis=0)/np.sum(thisCluster)
    #calculate the error value based on the center and it corresponding clusters
    for i in range (k):
        cluster1 = np.where(cluster==i,1,0)
        errorval += np.sum((((data*cluster1)[~np.all((data*cluster1)==0,axis=1)])-centers[i,:])**2)
    return centers, count, cluster, errorval


def main():
    #The main function
    errorlist=[]
    #k value
    k = 4
    #get the data
    data = dataprocessing()
    #get the returns value from the kmeans
    centers, iterations, indices, errorval = kmeans(data,k)
    #output
    print('My optimal value for k is this situation is 4')
    print('{} centers are: '.format(k), centers)
    print('The number of iterations is:', iterations)
    print('The final error value for k={} is'.format(k), errorval)
    print('The list labels (The center index of each point in the dataset, ranging from 0-3) is the following: \n', indices.tolist())
    #get the data for k from 1 to 10
    for i in range (1,11):
        c, it, ind, error = kmeans(data,i)
        kk = [i,error]
        errorlist.append(kk)
    errorlist[3][1] = errorval
    #put the data to an array for plotting
    errorlist = np.array(errorlist)
    return centers, data, errorlist

#output, print out the desired e centers, the number of iterations
#to convergence, the final error (sum of squared distances from points to centers), and
#labels (the assignment of points to centers). 
centers, data, err = main()

plt.scatter(np.array(centers)[:, 0], np.array(centers)[:, 1], color='red')
plt.scatter(data[:, 0], data[:, 1], alpha=0.1, color ='black')
plt.title('A graph of the original data and the centers ')
plt.show()

#error value of k clusters with k from 1 to 10
print(err)

#graph of error values of k clusters with k from 1 to 10
plt.plot(err[:, 0], err[:, 1], color='red')
plt.title('A graph of error vs k')
plt.show()