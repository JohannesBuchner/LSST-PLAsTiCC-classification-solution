from __future__ import print_function, division
from alltrain import *
from sklearn.metrics.classification import _weighted_sum

import matplotlib.pyplot as plt

def compare(A, B, novelmask):
	corr = (A * B).sum(axis=1)
	print(A.shape, B.shape, corr.shape)
	plt.hist(numpy.log10(corr), bins=1000, histtype='step')
	plt.hist(numpy.log10(corr[novelmask]), bins=1000, histtype='step')


#methods = 'KNN2', 'KNN4', 'RandomForest4'
methods = sys.argv[1:]

X = []
Z = []
Y = numpy.loadtxt('resampled_training_set_target.csv', dtype='i')

novel = numpy.loadtxt('test_set_all_sorted.csv.gz_novel_MM-IsolForest-0.04.csv', dtype='i') == -1

for method in methods:
	#print("loading %s ..." % (training_data_file + '_predictions_%s.csv.gz' % method))
	#X.append(pandas.read_csv(training_data_file + '_predictions_%s.csv.gz' % method, header=None).values)
	#if len(X) > 1:
	#	compare(X[0], X[-1])
	
	print("loading %s ..." % (unknown_data_file + '_predictions_%s.csv.gz' % method))
	Z.append(pandas.read_csv(unknown_data_file + '_predictions_%s.csv.gz' % method, header=None).values)
	if len(Z) > 1:
		compare(Z[0], Z[-1], novel)

plt.savefig('hyperpredict_difference.pdf', bbox_inches='tight')
plt.close()


