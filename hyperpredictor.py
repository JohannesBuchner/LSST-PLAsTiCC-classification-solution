from __future__ import print_function, division
from alltrain import *
from sklearn.metrics.classification import _weighted_sum

#methods = 'KNN2', 'KNN4', 'RandomForest4'
methods = sys.argv[1:]

X = []
Z = []
Y = numpy.loadtxt('training_set_target.csv', dtype='i')

for method in methods:
	print("loading %s ..." % (training_data_file + '_predictions_%s.csv.gz' % method))
	X.append(pandas.read_csv(training_data_file + '_predictions_%s.csv.gz' % method, header=None).values)
	
	print("loading %s ..." % (unknown_data_file + '_predictions_%s.csv.gz' % method))
	Z.append(pandas.read_csv(unknown_data_file + '_predictions_%s.csv.gz' % method, header=None).values)

X = numpy.hstack(tuple(X))
Z = numpy.hstack(tuple(Z))

# we want to train something that predicts T from A
sample_weight = weights_targets[Y]

def my_log_loss(y_true, y_pred, eps=1e-15, normalize=True, labels=None):
	transformed_labels = encoder.transform(y_true)
	y_pred = numpy.clip(y_pred, eps, 1 - eps)
	y_pred /= y_pred.sum(axis=1)[:, numpy.newaxis]
	
	sample_weight = weights_targets[y_true]
	loss = (transformed_labels * numpy.log(y_pred)).sum(axis=1)
	return _weighted_sum(loss, sample_weight, normalize)
	#return -(loss * weighting).sum() / weighting.sum()

scorer = make_scorer(my_log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)




t0 = time()
print("training MLP...")
clf = MLPClassifier(hidden_layer_sizes=40)
q = cross_val_score(clf, X, Y, cv=5, scoring=scorer)
print('%.3f +- %.3f' % (q.mean(), q.std()))
print('training done after %.1fs' % (time() - t0))

print('Predicting ...')
t0 = time()
print('  predictions for training data...')
predictions = cross_val_predict(clf, X, Y, cv=5, method='predict_proba')
print('    saving ...')
numpy.savetxt(training_data_file + '_hyperpredictions.csv.gz', predictions, delimiter=',', fmt='%.3e')
print('  predictions for unknown data...')
clf.fit(X, Y)
predictions = clf.predict_proba(Z)
print('    saving ...')
numpy.savetxt(unknown_data_file + '_hyperpredictions.csv.gz', predictions, delimiter=',', fmt='%.3e')
print('predictions done after %.1fs' % (time() - t0))



