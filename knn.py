import csv
import random
import math
import operator
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def loadDataset(filename, split=0.5, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(1, len(dataset)):
            for y in range(2):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def distance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def distanceSquared(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return distance


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = distanceSquared(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))  # no need to sort everything - enough to get first k positions
    neighbors = []
    for x in range(k):
        if x < len(distances):
            neighbors.append(distances[x][0])
    return neighbors


def getWeightedNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = distance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))  # no need to sort everything - enough to get first k positions
    neighbors = []
    for x in range(k):
        if x < len(distances):
            neighbors.append(distances[x])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    # print('sorted votes:', sortedVotes)
    return sortedVotes[0][0]


def getWeightedResponse(neighbors):  # assume neighbors are sorted
    classVotes = {}
    maxDist = neighbors[len(neighbors) - 1][1]
    minDist = neighbors[0][1]

    for x in range(len(neighbors)):

        neighborWeight = 1
        if minDist != maxDist:
            neighborWeight = (maxDist - neighbors[x][1]) / (maxDist - minDist)

        response = neighbors[x][0][-1]
        if response in classVotes:
            classVotes[response] += neighborWeight
        else:
            classVotes[response] = neighborWeight

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def runPrediction(trainingSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        if x % 10 == 0:
            sys.stdout.write("\r%.2f%%" % (100.0*x/float(len(testSet))))
            sys.stdout.flush()
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
    sys.stdout.write("\r")
    sys.stdout.flush()
    return predictions


def runWeightedPrediction(trainingSet, testSet, k):
    predictions = []
    for x in range(len(testSet)):
        if x % 10 == 0:
            sys.stdout.write("\r%.2f%%" % (100.0*x/float(len(testSet))))
            sys.stdout.flush()
        neighbors = getWeightedNeighbors(trainingSet, testSet[x], k)
        result = getWeightedResponse(neighbors)
        predictions.append(result)
    sys.stdout.write("\r")
    sys.stdout.flush()
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    if float(len(testSet)) > 0:
        return (correct / float(len(testSet))) * 100.0
    return 100.0


def drawPlots(trainingSet, testSet, predictions, k, dataFilename):
    fig = plt.figure(figsize=(8, 8))

    # generate test set with linspace

    sortedX = sorted(trainingSet+testSet, key=operator.itemgetter(0))
    sortedY = sorted(trainingSet+testSet, key=operator.itemgetter(1))

    genX = np.linspace(sortedX[0][0], sortedX[len(trainingSet+testSet)-1][0], 100)
    genY = np.linspace(sortedY[0][1], sortedY[len(trainingSet+testSet)-1][1], 100)
    genTestSet = [[genX[i], genY[j], -1] for i in range(len(genX)) for j in range(len(genY))]
    genPredictions = runPrediction(trainingSet, genTestSet, k)

    gx = [genTestSet[i][0] for i in range(len(genTestSet))]
    gy = [genTestSet[i][1] for i in range(len(genTestSet))]
    gcolors = []
    for i, testInstance in enumerate(genTestSet):
        if genPredictions[i] == '1':
            gcolors.append('deepskyblue')
        elif genPredictions[i] == '2':
            gcolors.append('lightcoral')
        else:
            gcolors.append('springgreen')
    plt.scatter(gx, gy, c=gcolors, alpha=0.2)


    x = [trainingSet[i][0] for i in range(len(trainingSet))]
    y = [trainingSet[i][1] for i in range(len(trainingSet))]
    colors = []
    for i, trainingInstance in enumerate(trainingSet):
        if trainingInstance[2] == '1':
            colors.append('blue')
        elif trainingInstance[2] == '2':
            colors.append('red')
        else:
            colors.append('green')    
    plt.scatter(x, y, c=colors)

    x1 = [testSet[i][0] for i in range(len(testSet))]
    y1 = [testSet[i][1] for i in range(len(testSet))]
    colors1 = []
    for i, testInstance in enumerate(testSet):
        if predictions[i] == '1':
            colors1.append('deepskyblue')
        elif predictions[i] == '2':
            colors1.append('lightcoral')
        else:
            colors1.append('springgreen')
    plt.scatter(x1, y1, c=colors1, marker='^', alpha=1.0)

    plt.savefig(dataFilename + '_scatterplot' + '.png', bbox_inches='tight')
    plt.show()
    return


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.6
    k = 5
    dataFilename = 'data.three_gauss.train.100'
    loadDataset('Data/' + dataFilename + '.csv', split, trainingSet, testSet)  # timeit
    predictions = runPrediction(trainingSet, testSet, k)
    #predictions = runWeightedPrediction(trainingSet, testSet, k)
    accuracy = getAccuracy(testSet, predictions)
    print("Accuracy: %.2f%%" % round(accuracy, 2))
    drawPlots(trainingSet, testSet, predictions, k, dataFilename)


main()
