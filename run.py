import knn
import weighted_knn


knn_instance = knn.Knn()
accuracy = knn_instance.run(
	k=15, split=0.6, dataFilename='data.simple.train.100', scatterPlotFilename='data.simple.train.100', withGeneratedTestSet=True)
print("Accuracy: %.2f%%" % round(accuracy, 2))

weightedKnn_instance = weighted_knn.WeightedKnn()
accuracy = weightedKnn_instance.run(
	k=15, split=0.6, dataFilename='data.three_gauss.train.100', scatterPlotFilename='data.simple.train.100', withGeneratedTestSet=True)
print("Accuracy: %.2f%%" % round(accuracy, 2))