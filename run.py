import knn
import weighted_knn
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


if not os.path.exists('Plots'):
    os.makedirs('Plots')

split = 0.6
nrOfExperiments = 1
kList = list(range(3, 6, 1))

for dataset in ['simple', 'three_gauss']:
	for datasize in [100, 500]:
		print('Bare k-NN:')

		knn_instance = knn.Knn()
		accuracies = {}
		averageAccuracies1 = {}
		for k in kList:
			accuracies[k] = []
			for i in range(nrOfExperiments):
				print('Running experiment nr ' + repr(i + 1) + '/' + repr(nrOfExperiments) + ' for k = ' + repr(k))
				accuracy = knn_instance.run(
					k, 
					split=split, 
					dataFilename='data.' + dataset + '.train.' + repr(datasize), 
					scatterPlotFilename='bare_simple_' + repr(datasize), 
					withGeneratedTestSet=True)
				print("Accuracy: %.2f%%" % round(accuracy, 2))
				accuracies[k].append(accuracy)
			averageAccuracies1[k] = float(sum(accuracies[k]))/float(nrOfExperiments)
			print('Average accuracy: ' + repr(round(averageAccuracies1[k], 2)) + '%')


		print('Weighted k-NN:')

		weightedKnn_instance = weighted_knn.WeightedKnn()
		accuracies = {}
		averageAccuracies2 = {}
		for k in kList:
			accuracies[k] = []
			for i in range(nrOfExperiments):
				print('Running experiment nr ' + repr(i + 1) + '/' + repr(nrOfExperiments) + ' for k = ' + repr(k))
				accuracy = weightedKnn_instance.run(
					k, 
					split=split, 
					dataFilename='data.' + dataset + '.train.' + repr(datasize), 
					scatterPlotFilename='weighted_simple_' + repr(datasize), 
					withGeneratedTestSet=True)
				print("Accuracy: %.2f%%" % round(accuracy, 2))
				accuracies[k].append(accuracy)
			averageAccuracies2[k] = float(sum(accuracies[k]))/float(nrOfExperiments)
			print('Average accuracy: ' + repr(round(averageAccuracies2[k], 2)) + '%')


		# draw plots

		fig = plt.figure(figsize=(8, 8))

		lists1 = sorted(averageAccuracies1.items())
		x1, y1 = zip(*lists1)
		plt.plot(x1, y1, 'ro-')

		lists2 = sorted(averageAccuracies2.items())
		x2, y2 = zip(*lists2)
		plt.plot(x2, y2, 'bo-')

		plt.xlabel('k')
		plt.ylabel('accuracy %')

		#plt.show()
		plt.savefig('Plots/' + dataset + repr(datasize) + '_AccuraciesPlot' + '.png')
		plt.close(fig)