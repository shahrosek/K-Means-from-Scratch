import pandas as pand
import numpy as np
import random as rand
import pylab as P
from collections import defaultdict

kID = 1

def processFile(file):
    dataframe = pand.read_table(file, sep='\n', header=None)
    data = np.array(dataframe)
    return data

def initCentroids(k, data):
    index = rand.sample(range(0, data.shape[0]), k)
    initialCentroids = data[index]
    return initialCentroids

def calculateEuclidean(pt1,pt2):
    return np.linalg.norm(pt1 - pt2)

def initClusters(centroids, data):
    clusters = dict()
    for value in data:
        distances = []
        for cent in centroids:
            dist = calculateEuclidean(cent, value)
            distances.append(dist)
        nearestIndex = np.argmin(distances)
        nearestCentroid = tuple(centroids[nearestIndex])
        if nearestCentroid not in clusters:
            clusters[nearestCentroid] = []
        clusters[nearestCentroid].append(value)
    return clusters

def runKMeans(k, data, maxIter):
    instances, features = data.shape
    centroids = initCentroids(k, data)
    finalClusters = dict()
    i=0
    for index in range(0, maxIter):
        clusters = initClusters(centroids, data)
        finalClusters = clusters
        updatedCentroids = []
        for centroid, clusterValues in clusters.items():
            updatedCentroid = np.mean(clusterValues, axis=0)
            updatedCentroids.append(updatedCentroid)
        if calculateEuclidean(np.array(updatedCentroids), centroids) == 0:
            i += 1
        if i == 3:
            break
        centroids = updatedCentroids

    distCost = distortionCost(finalClusters, instances)

    return distCost, finalClusters

def assignClusterID(clusters):
    global kID
    i=1
    file = open("Cluster"+str(kID)+".txt", "w")
    for centroid, values in clusters.items():
        for value in values:
            file.write(str(centroid)+" "+str(value)+" "+str(i)+"\n")
        i+=1
    file.close()
    kID += 1

def distortionCost(clusters, instances):
    cost = 0
    for centroid, clusterValues in clusters.items():
        for value in clusterValues:
            eucDist = calculateEuclidean(value, centroid)
            squaredEucDist = eucDist**2
            cost += squaredEucDist
    return cost/instances

def plot_clusters(clusters, x_axis_label="", y_axis_label="", plot_title="Optimal K Means Clusters"):
    list_of_colors = ['firebrick', 'gold', 'navy', 'lightseagreen', 'deepskyblue', 'mediumpurple',
                          'darkmagenta', 'palevioletred', 'darkgreen',  'darkorange', 'darkslategray', 'dimgrey']
    color_index = 0
    for centroid, cluster_points in clusters.items():
        cluster_color = list_of_colors[color_index]
        x_values_index = 0
        y_values_index = 1

        P.scatter(centroid[x_values_index], centroid[y_values_index], color=cluster_color, s=500, alpha=0.5)

        cluster_points_x_values = [cluster_point[x_values_index] for cluster_point in cluster_points]
        cluster_points_y_values = [cluster_point[y_values_index] for cluster_point in cluster_points]

        P.scatter(cluster_points_x_values, cluster_points_y_values, color=cluster_color, s=100, marker='o')
        color_index += 1

    P.title(plot_title)
    P.xlabel(x_axis_label)
    P.ylabel(y_axis_label)
    P.show()

def main():
    numCentroids = int(input("Enter number of centroids: "))
    dataset = processFile('Cluster4.txt')

    dCosts = []
    allClusters = []
    K = range(1,numCentroids+1)

    for k in K:
        distCost, clusters = runKMeans(k, dataset, maxIter= 20)
        dCosts.append(distCost)
        allClusters.append(clusters)
        assignClusterID(clusters)

    plot_clusters(allClusters[3])

    dCosts = np.array(dCosts)
    P.plot(K, dCosts, 'bx-')
    P.xlabel('K')
    P.ylabel('Distortion')
    P.title('The Elbow Method showing the optimal K')
    P.show()
    # plot_clusters(allClusters[:-1])

if __name__=="__main__":
    main()
