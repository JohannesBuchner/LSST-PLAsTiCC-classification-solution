from __future__ import print_function, division
from alltrain import *

# for random forests the transformation should not matter.
#qt = QuantileTransformer()
#X = qt.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')        
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	#unknown = qt.transform(unknown)

def train_and_evaluate(name, clf):
	print()
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	q = cross_val_score(clf, X, Y, cv=4, scoring=scorer, n_jobs=4)
	print('%2.2f +- %2.2f %s (training speed: %.1fs)' % (q.mean(), q.std(), name, time() - t0))

	if not execute:
		# manual masking of one third
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 21)
		clf = clf.fit(X_train, y_train)
		t0 = time()
		for i in range(10):
			y_pred = clf.predict(X_test)
		
		#y_scores = clf.predict_proba(X_ts)
		if hasattr(clf, 'feature_importances_') and False:
			importances = clf.feature_importances_
			indices = numpy.argsort(importances)[::-1]
			#std = numpy.std([entity.feature_importances_ for entity in clf.estimators_], axis=0)

			# Print the feature ranking
			print("Feature ranking:")

			for f, index in enumerate(indices):
				print("%d. feature %d (%f) -- %s" % (f + 1, index, importances[index], train.columns[index]))
				if importances[indices[f]] < importances[indices[0]] / 100.0:
					break

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
		predictions = cross_val_predict(clf, X, Y, cv=4, method='predict_proba', n_jobs=4)
		numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',', fmt='%.4e')
		clf.fit(X, Y)
		predictions = clf.predict_proba(unknown)
		print('predictions for unknown data...')
		numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',', fmt='%.4e')
		print('predictions done after %.1fs' % (time() - t0))
	return clf


if os.environ.get('FIND_FEATURE_SUBSET', '0') == '1':
	clf = RandomForestClassifier(n_estimators=100)
	from sklearn.feature_selection import RFE, RFECV
	from sklearn.model_selection import StratifiedKFold
	print("Recursive feature elimination with RandomForest100...")
	t0 = time()
	rfe = RFE(estimator=clf, n_features_to_select=40, step=0.1, verbose=2)
	#rfe = RFECV(estimator=clf, min_features_to_select=10, step=0.1, verbose=2, 
	#	cv=StratifiedKFold(2), scoring=scorer)
	rfe.fit(X, Y)
	print('done after %.1fs' % (time() - t0))
	print(rfe.grid_scores_)

	indices = numpy.argsort(rfe.ranking_)
	print("Feature ranking:")
	fcols = open('important_columns.txt', 'w')
	for f, index in enumerate(indices):
		print("%d. feature %d (%d) -- %s" % (f + 1, index, rfe.ranking_[index], train.columns[index]))
		if rfe.ranking_[index] == 1:
			fcols.write("%d\t%s\n" % (index,train.columns[index]))
	
	sys.exit(0)

#train_and_evaluate('KNN2', clf = KNeighborsClassifier(n_neighbors=2))
#train_and_evaluate('KNN4', clf = KNeighborsClassifier(n_neighbors=4))
#train_and_evaluate('KNN10', clf = KNeighborsClassifier(n_neighbors=10))
#train_and_evaluate('KNN40', clf = KNeighborsClassifier(n_neighbors=40))

#train_and_evaluate('XGradientBoosting', clf = XGBClassifier(n_estimators=40))
train_and_evaluate('RandomForest4', clf = RandomForestClassifier(n_estimators=4))
train_and_evaluate('RandomForest10', clf = RandomForestClassifier(n_estimators=10))
train_and_evaluate('RandomForest40', clf = RandomForestClassifier(n_estimators=40))
train_and_evaluate('RandomForest100', clf = RandomForestClassifier(n_estimators=100))
train_and_evaluate('RandomForest400', clf = RandomForestClassifier(n_estimators=400))
train_and_evaluate('AdaBoost40', clf = AdaBoostClassifier(n_estimators=40))
train_and_evaluate('AdaBoost400', clf = AdaBoostClassifier(n_estimators=400))
train_and_evaluate('ExtraTrees40', clf = ExtraTreesClassifier(n_estimators=40))

# TODO:

# sklearn.discriminant_analysis.LinearDiscriminantAnalysis
# sklearn.neighbors.NearestCentroid

## too slow
# train_and_evaluate('GradientBoosting', clf = GradientBoostingClassifier(n_estimators=10)) # extremely slow

## can't predict probabilities
#train_and_evaluate('RidgeClassifier', clf = RidgeClassifier())

## too slow
#train_and_evaluate('SVC-default', clf = SVC(probability=True))
#train_and_evaluate('SVC-0.1', clf = SVC(probability=True, C = 0.1, gamma = 0.05))
#train_and_evaluate('LinearSVC-default', clf = LinearSVC(probability=True))
#train_and_evaluate('LinearSVC-0.1', clf = LinearSVC(probability=True, C = 0.1, gamma = 0.05))


