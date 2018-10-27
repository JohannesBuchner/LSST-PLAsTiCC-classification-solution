from __future__ import print_function, division
from alltrain import *
from numpy import exp, pi, sqrt, log10, log
import matplotlib.pyplot as plt

columns = train.columns[valid_column_mask]
del train

#X = mytransformer.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')
	unknown = unknown.values
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)
	#unknown = mytransformer.transform(unknown)
else:
	unknown = None

from naivebayes import MultiGaussNaiveBayesClassifier

name = 'GaussianNaiveBayes'
all_classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]

Ncovered = len(labels)
print('%d columns, %d classes, %d training samples' % (X.shape[1], Ncovered, len(X)))
t0 = time()
print()
print('running %s ...' % name)

cls = MultiGaussNaiveBayesClassifier(all_labels=all_classes, 
	plot_prefix = 'viz_NaiveBayes_', column_names = columns,
	noteworthy_information=10, verbose=1)
cls.fit(X, Y)
Xpredict = cls.predict_proba(X)
lowest_prob = cls.surprise_.min()
Xpredict[:,-1] = exp(cls.surprise_ - lowest_prob)
print('worst surprise:', Xpredict[:,-1].min())
if execute:
	Ypredict = cls.predict_proba(unknown)
	Ypredict[:,-1] = exp(cls.surprise_ - lowest_prob)

if execute:
	print('predictions for training data...')
	numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, Xpredict, delimiter=',', fmt='%.4e')
	print('predictions for unknown data...')
	numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, Ypredict, delimiter=',', fmt='%.4e')
	print('predictions done after %.1fs' % (time() - t0))


