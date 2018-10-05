from __future__ import print_function, division
from alltrain import *

qt = QuantileTransformer()
X = qt.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')        
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	unknown = qt.transform(unknown)

def train_and_evaluate(name, clf):
	print()
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	q = cross_val_score(clf, X_white, Y, cv=5, scoring=scorer)
	print('%2.2f +- %2.2f %s (training speed: %.1fs)' % (q.mean(), q.std(), name, time() - t0))

	if not execute:
		# manual masking of one third
		X_train, X_test, y_train, y_test = train_test_split(X_white, Y, test_size = 0.25, random_state = 21)
		clf = clf.fit(X_train, y_train)
		t0 = time()
		for i in range(10):
			y_pred = clf.predict(X_test)
		
		print('confusion matrix: (eval speed: %.2fs)' % (time() - t0))
	#cnf_matrix = confusion_matrix(y_test, y_pred)
	#print(cnf_matrix)
	#print 'ROC curve plot...'
	#fpr, tpr, thresholds = roc_curve(y_ts, y_scores[:,1])
	#plt.title(name)
	##print fpr, tpr, thresholds
	#print '5% FPR: at threshold', thresholds[fpr < 0.05][-1], 'with efficiency', tpr[fpr < 0.05][-1]*100, '%'
	#print '1% FPR: at threshold', thresholds[fpr < 0.01][-1], 'with efficiency', tpr[fpr < 0.01][-1]*100, '%'
	#plt.plot(fpr, tpr, '-', color='r')
	#plt.plot([0,1], [0,1], ':', color='k')
	#plt.xlabel('False positive rate')
	#plt.ylabel('True positive rate')
	#plt.savefig('trainactivitydetect_scores_%s.pdf' % name, bbox_inches='tight')
	#plt.close()
	if execute:
		t0 = time()
		print('predictions for training data...')
		predictions = cross_val_predict(clf, X_white, Y, method='predict_proba')
		numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',', fmt='%.4e')
		clf.fit(X_white, Y)
		predictions = clf.predict_proba(unknown_white)
		print('predictions for unknown data...')
		numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',', fmt='%.4e')
		print('predictions done after %.1fs' % (time() - t0))
	return clf

for n_components in 10, 40, 100:
	print("dimensionality reduction with PCA-%d" % n_components)
	prefix = 'PCA-%d' % n_components
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
	print("done in %0.3fs" % (time() - t0))
	X_white = pca.transform(X)
	if execute:
		unknown_white = pca.transform(unknown)

	#train_and_evaluate(prefix + 'KNN2', clf = KNeighborsClassifier(n_neighbors=2))
	#train_and_evaluate(prefix + 'KNN4', clf = KNeighborsClassifier(n_neighbors=4))
	#train_and_evaluate(prefix + 'KNN10', clf = KNeighborsClassifier(n_neighbors=10))
	#train_and_evaluate(prefix + 'KNN40', clf = KNeighborsClassifier(n_neighbors=40))
	#train_and_evaluate('KNN40', clf = KNeighborsClassifier(n_neighbors=40))
	#train_and_evaluate('KNN10-r', clf = KNeighborsClassifier(n_neighbors=10, weights='distance'))
	#train_and_evaluate('KNN40-r', clf = KNeighborsClassifier(n_neighbors=40, weights='distance'))

	#train_and_evaluate(prefix + 'RandomForest4', clf = RandomForestClassifier(n_estimators=4))
	#train_and_evaluate(prefix + 'RandomForest10', clf = RandomForestClassifier(n_estimators=10))
	#train_and_evaluate(prefix + 'RandomForest40', clf = RandomForestClassifier(n_estimators=40))
	#train_and_evaluate(prefix + 'RandomForest100', clf = RandomForestClassifier(n_estimators=100))
	#train_and_evaluate(prefix + 'RandomForest400', clf = RandomForestClassifier(n_estimators=400))
	#train_and_evaluate(prefix + 'RandomForest100', clf = RandomForestClassifier(n_estimators=100))
	#train_and_evaluate(prefix + 'AdaBoost', clf = AdaBoostClassifier(n_estimators=40))
	#train_and_evaluate(prefix + 'GradientBoosting', clf = GradientBoostingClassifier(n_estimators=40))
	#train_and_evaluate(prefix + 'ExtraTrees', clf = ExtraTreesClassifier(n_estimators=40))
	
	train_and_evaluate(prefix + 'MLP4', clf = MLPClassifier(hidden_layer_sizes=(4,)))
	train_and_evaluate(prefix + 'MLP10', clf = MLPClassifier(hidden_layer_sizes=(10,)))
	train_and_evaluate(prefix + 'MLP10-20-10', clf = MLPClassifier(hidden_layer_sizes=(10,20,10)))
	train_and_evaluate(prefix + 'MLP4-16-4', clf = MLPClassifier(hidden_layer_sizes=(4,16,4)))
	train_and_evaluate(prefix + 'MLP40', clf = MLPClassifier(hidden_layer_sizes=(40,)))

	#train_and_evaluate(prefix + 'LinearSVC-default', clf = LinearSVC())
	#train_and_evaluate(prefix + 'LinearSVC-0.1', clf = LinearSVC(C = 0.1, gamma = 0.05))
	#train_and_evaluate(prefix + 'SVC-default', clf = SVC(probability=True))
	#train_and_evaluate(prefix + 'SVC-0.1', clf = SVC(probability=True, C = 0.1, gamma = 0.05))
	
	#break


