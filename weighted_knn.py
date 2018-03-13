import csv
import random
import math
import operator
import sys
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import knn

class WeightedKnn(knn.Knn):
    def distance(self, instance1, instance2, length):
        """euclidean distance"""
        distance = 0
        for x in range(length):
            distance += pow((instance1[x] - instance2[x]), 2)
        return math.sqrt(distance)


    def getResponse(self, neighbors):
        """distance-weighted vote"""
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