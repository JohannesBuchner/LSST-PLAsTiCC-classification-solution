from __future__ import print_function, division
from alltrain import *
from sklearn.metrics.classification import _weighted_sum

#methods = 'KNN2', 'KNN4', 'RandomForest4'
methods = sys.argv[1:]

A = []
B = []
Y = numpy.loadtxt('training_set_target.csv', dtype='i')

for method in methods:
	print("loading %s ..." % (training_data_file + '_predictions_%s.csv.gz' % method))
	A.append(pandas.read_csv(training_data_file + '_predictions_%s.csv.gz' % method, header=None).values)
	
	print("loading %s ..." % (unknown_data_file + '_predictions_%s.csv.gz' % method))
	B.append(pandas.read_csv(unknown_data_file + '_predictions_%s.csv.gz' % method, header=None).values)

X = numpy.hstack(tuple(A))
Z = numpy.hstack(tuple(B))

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
q = cross_val_score(clf, X, Y, cv=4, scoring=scorer)
print(q.mean(), q.std())
print('training done after %.1fs' % (time() - t0))

t0 = time()
print('predictions for training data...')
predictions = cross_val_predict(clf, X, Y, method='predict_proba')
numpy.savetxt(training_data_file + '_hyperpredictions.csv.gz', predictions, delimiter=',')
clf.fit(X, Y)
predictions = clf.predict_proba(Z)
print('predictions for unknown data...')
numpy.savetxt(unknown_data_file + '_hyperpredictions.csv.gz', predictions, delimiter=',')
print('predictions done after %.1fs' % (time() - t0))



