import csv
import math
import operator
import numpy as np

# IMPORTANT: see README.MD for additional project documentation including:
# installation requirements, data quality report and pre-processing steps

# Import the breast cancer data into a csv
def getData():
    with open('./data/breast-cancer-wisconsin2.data', 'rb') as myfile:
        df = list(csv.reader(myfile))

    for x in range(len(df)):
        del df[x][0]  # Remove the patient ID from the test data
        for y in range(10):
            df[x][y] = float(df[x][y])  #convert datatype to float
    return df

# Calculate the euclidean distance between two instances
def getDistance(d1, d2, length): 
    distance = 0
    for x in range(length):
        distance += pow((d1[x] - d2[x]), 2)
    return math.sqrt(distance)

# Get the k nearest neighbours to the newInput which
# in our case is the test dataset
def getNeighbors(trainSet, newInput, k):
    distances = []
    length = len(newInput)-1

    for x in range(len(trainSet)):
        dist = getDistance(newInput, trainSet[x], length)
        distances.append((trainSet[x], dist))

    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

# Tally the most common class prediction from the returned
# set of neighbors and take the one with the most votes
def getClass(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.iteritems(),
                         key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

# Calculate the accuracy of our predictions by comparing
# them to our labeled data
def getPerformance(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0

# 
def runExperiment():

    # Gather the dataset from the txt file
    dataSet = getData()
    k = 3   #number of nearest neighbours
    folds = 10  #number of cross-validation folds

    print "Data set rows: ",len(dataSet)
    print "K nearest neighbors:", k
    print "Cross-validation folds:", folds
    print "-----------------------------"

    # Iterate through each value of k
    for z in range(k):
        
        totalAccuracy = 0.0

        # Iterate through the number of cross-validation folds
        for f in range(folds):
            testSet = []
            trainSet = []
            
            # Iterate through each row in the dataset
            for r in range(len(dataSet)):
              ind = r % 10
              if(ind == f):
                  testSet.append(dataSet[r]) #assign 1/10 to test dataset
              else:
                  trainSet.append(dataSet[r]) #assign 9/10 to training dataset

            predictions = []

            # Iterate through each row in the test dataset
            for x in range(len(testSet)):
              neighbors = getNeighbors(trainSet, testSet[x], z+1)
              result = getClass(neighbors)
              predictions.append(result)
              #uncomment line below if you wish to see the predictions
              #print('predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
            accuracy = getPerformance(testSet, predictions)
            #uncomment the line below if you wish to see each folds accuracy
            #print "Fold",f+1,"Accuracy: " + repr(accuracy) + "%"
            totalAccuracy += accuracy

        avgAccuracy = (totalAccuracy / folds)
        print "K=",z+1,"Accuracy: " + repr(avgAccuracy) + "%"
    print ""

runExperiment()