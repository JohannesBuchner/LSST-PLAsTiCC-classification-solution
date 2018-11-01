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
	unknown_object_ids = unknown.pop('object_id').values
	unknown = unknown.values
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)
	#unknown = mytransformer.transform(unknown)
else:
	unknown = None

from naivebayes import MultiGaussNaiveBayesClassifier
import joblib

name = 'GaussianNaiveBayes'
all_classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]

Ncovered = len(labels)
print('%d columns, %d classes, %d training samples' % (X.shape[1], Ncovered, len(X)))
t0 = time()
print()
print('running %s ...' % name)

clf = MultiGaussNaiveBayesClassifier()
q = cross_val_score(clf, X, Y, cv=4, scoring=scorer, n_jobs=4)
print('%2.2f +- %2.2f %s (CV training speed: %.1fs)' % (q.mean(), q.std(), name, time() - t0))

clf = MultiGaussNaiveBayesClassifier(all_labels=all_classes, 
	column_names = columns, plot_prefix = 'viz_NaiveBayes_',
	noteworthy_information=10, verbose=1, parallel=joblib.Parallel(n_jobs=-1))

t0 = time()
clf.fit(X, Y)
print('training speed: %.1fs' % (time() - t0))
Xpredict = clf.predict_proba(X)
lowest_prob = clf.surprise_.min()
print('worst prob in train:', lowest_prob)
#Xpredict[:,-1] = numpy.where(clf.surprise_ > lowest_prob, 0, exp(clf.surprise_ - lowest_prob))
Xsurprise = clf.surprise_
#Xpredict[:,-1] = clf.surprise_
if execute:
	Ypredict = clf.predict_proba(unknown)
	#Ypredict[:,-1] = numpy.where(clf.surprise_ > lowest_prob, 0, exp(clf.surprise_ - lowest_prob))
	#Ypredict[:,-1] = clf.surprise_
	Ysurprise = clf.surprise_
	print('worst prob in test:', clf.surprise_.min())

if execute:
	print('predictions for training data...')
	write_prediction(training_data_file + '_predictions_%s.csv.gz' % name, training_object_ids, Xpredict, outlierproba=Xsurprise)
	print('predictions for unknown data...')
	write_prediction(unknown_data_file + '_predictions_%s.csv.gz' % name, unknown_object_ids, Ypredict, outlierproba=Ysurprise)
	print('predictions done after %.1fs' % (time() - t0))


