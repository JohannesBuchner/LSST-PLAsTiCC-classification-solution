from __future__ import print_function, division
from alltrain import *
from sklearn.metrics.classification import _weighted_sum

def read_csv(filename):
	data = pandas.read_csv(filename)
	object_ids = data.pop('object_id').values
	data.pop('class_99')
	prob = data.values
	prob /= prob.sum(axis=1).reshape((-1,1))
	return object_ids, prob

#methods = 'KNN2', 'KNN4', 'RandomForest4'
methods = sys.argv[1:]

X = []
Z = []
#Y = numpy.loadtxt('resampled_training_set_target.csv', dtype='i')
Y = Y_orig
Xsums = 0
Zsums = 0
Zsurprise = 0

for i, method in enumerate(methods):
	print("loading %s ..." % (training_data_file + '_predictions_%s.csv.gz' % method))
	training_object_ids, Xi = read_csv(training_data_file + '_predictions_%s.csv.gz' % method)
	Xsums = Xsums + Xi
	X.append(Xi)
	del Xi
	
	print("loading %s ..." % (unknown_data_file + '_predictions_%s.csv.gz' % method))
	unknown_object_ids, Zi = read_csv(unknown_data_file + '_predictions_%s.csv.gz' % method)
	Zsums = Zsums + Zi
	Z.append(Zi)
	del Zi

varX = numpy.var(X, axis=0).mean(axis=1)
varZ = numpy.var(Z, axis=0).mean(axis=1)
X = numpy.hstack(tuple(X))
Z = numpy.hstack(tuple(Z))


N = int(os.environ.get('NNEURONS', '40'))
t0 = time()
method = os.environ['METHOD']
if method == 'AVG':
	write_prediction(training_data_file + '_hyperpredictions-%s.csv.gz' % method, training_object_ids, Xsums / len(methods), outlierproba=varX)
	write_prediction(unknown_data_file + '_hyperpredictions-%s.csv.gz' % method, unknown_object_ids, Zsums / len(methods), outlierproba=varZ)
	sys.exit(0)
	
elif method == 'MLP':
	print("training MLP...")
	clf = MLPClassifier(hidden_layer_sizes=N, max_iter=2000)
elif method == 'RF':
	print("training RF...")
	clf = RandomForestClassifier(n_estimators=400, class_weight=class_weights)
elif method == 'SVC':
	print("training SVC...")
	clf = SVC(probability=True, class_weight=class_weights, gamma='auto')
q = cross_val_score(clf, X, Y, cv=4, scoring=scorer, n_jobs=4)
print('%.3f +- %.3f' % (q.mean(), q.std()))
print('training done after %.1fs' % (time() - t0))

from sklearn.metrics import confusion_matrix
print("Confusion matrix:")
predictions = cross_val_predict(clf, X, Y, cv=4, n_jobs=4)
cnf_matrix = confusion_matrix(Y, predictions)
print('  '.join(['%3s' % l for l in [''] + list(labels)]))
for l, cnf_row in zip(labels, cnf_matrix):
	print('  '.join(['%3s' % l] + ['%3d' % cnf_cell for cnf_cell in cnf_row]))

cnf_matrix = (cnf_matrix * 100 / N_labels.reshape((-1,1))).astype(int)
print("Confusion matrix, normalised:")
print('  '.join(['%3s' % l for l in [''] + list(labels)]))
for l, cnf_row in zip(labels, cnf_matrix):
	print('  '.join(['%3s' % l] + ['%3d' % cnf_cell for cnf_cell in cnf_row]))

print("Confusion examples:")
for l, cnf_row in zip(labels, cnf_matrix):
	for l2, cnf_cell in zip(labels, cnf_row):
		if l == l2 or cnf_cell < 10: 
			continue
		mask = numpy.logical_and(Y == l, l2 == predictions)
		print("%s confused as %s:" % (l, l2), 
			",".join(['%d' % i for i in training_object_ids[mask][:4]]))

print('Predicting ...')
t0 = time()
print('  predictions for training data...')
predictions = cross_val_predict(clf, X, Y, cv=4, method='predict_proba', n_jobs=4)
print('    saving ...')
write_prediction(training_data_file + '_hyperpredictions-%s.csv.gz' % method, training_object_ids, predictions)
print('  predictions for unknown data...')
clf.fit(X, Y)
predictions = clf.predict_proba(Z)
print('    saving ...')
write_prediction(unknown_data_file + '_hyperpredictions-%s.csv.gz' % method, unknown_object_ids, predictions)
print('predictions done after %.1fs' % (time() - t0))



