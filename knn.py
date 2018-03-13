import csv
import random
import math
import operator
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

class Knn:
    def loadDataset(self, filename, split=0.5, trainingSet=[], testSet=[]):
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


    def distance(self, instance1, instance2, length):
        """squared euclidean distance because it's only used in sorting"""
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return distance


    def getNeighbors(self, trainingSet, testInstance, k):
        distances = []
        length = len(testInstance) - 1
        for x in range(len(trainingSet)):
            dist = self.distance(testInstance, trainingSet[x], length)
            distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))  # no need to sort everything - enough to get first k positions

        neighbors = []
        for x in range(k):
            if x < len(distances):
                neighbors.append(distances[x])
        return neighbors


    def getResponse(self, neighbors):
        """majority vote"""
        classVotes = {}
        for x in range(len(neighbors)):
            response = neighbors[x][0][-1]
            if response in classVotes:
                classVotes[response] += 1
            else:
                classVotes[response] = 1

        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
        return sortedVotes[0][0]


    def printProgress(self, text, percent):
        sys.stdout.write("\r" + text + " progress: %.2f%%" % percent)
        sys.stdout.flush()

    def runPrediction(self, trainingSet, testSet, k):
        predictions = []
        for x in range(len(testSet)):
            if x % 10 == 0:
                sys.stdout.write("\rPrediction progress: %.2f%%" % (100.0*x/float(len(testSet))))
                sys.stdout.flush()
            neighbors = self.getNeighbors(trainingSet, testSet[x], k)
            result = self.getResponse(neighbors)
            predictions.append(result)

        sys.stdout.write("\r\033[K")
        sys.stdout.flush()
        return predictions


    def getAccuracy(self, testSet, predictions):
        correct = 0
        for x in range(len(testSet)):
            if testSet[x][-1] == predictions[x]:
                correct += 1

        if float(len(testSet)) > 0:
            return (correct / float(len(testSet))) * 100.0
        return 100.0


    def drawPlots(self, trainingSet, testSet, predictions, k, scatterPlotFilename, withGeneratedTestSet):
        fig = plt.figure(figsize=(8, 8))

        # generate test set with linspace

        if withGeneratedTestSet:
            sortedX = sorted(trainingSet+testSet, key=operator.itemgetter(0))
            sortedY = sorted(trainingSet+testSet, key=operator.itemgetter(1))

            genX = np.linspace(sortedX[0][0], sortedX[len(trainingSet+testSet)-1][0], 100)
            genY = np.linspace(sortedY[0][1], sortedY[len(trainingSet+testSet)-1][1], 100)
            genTestSet = [[genX[i], genY[j], -1] for i in range(len(genX)) for j in range(len(genY))]
            genPredictions = self.runPrediction(trainingSet, genTestSet, k)

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

        plt.savefig('Plots/' + scatterPlotFilename + '.png', bbox_inches='tight')
        #plt.show()
        return


    def run(self, k, split, dataFilename, scatterPlotFilename, withGeneratedTestSet=False):
        trainingSet, testSet = [], []
        self.loadDataset('Data/' + dataFilename + '.csv', split, trainingSet, testSet)
        predictions = self.runPrediction(trainingSet, testSet, k)
        accuracy = self.getAccuracy(testSet, predictions)
        self.drawPlots(trainingSet, testSet, predictions, k, scatterPlotFilename, withGeneratedTestSet)
        return accuracy