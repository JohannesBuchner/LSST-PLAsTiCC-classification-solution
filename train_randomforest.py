from __future__ import print_function, division
from alltrain import *

train_columns = list(train.columns[valid_column_mask])
del train

# for random forests the transformation should not matter.
#qt = QuantileTransformer()
#X = qt.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id').values
	unknown = unknown.values
	print('unknown:', unknown.shape)
	unknown = imp.transform(unknown)
	#unknown[~numpy.isfinite(unknown)] = -99
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
				print("%d. feature %d (%f) -- %s" % (f + 1, index, importances[index], train_columns[index]))
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
		write_prediction(training_data_file + '_predictions_%s.csv.gz' % name, training_object_ids, predictions)
		clf.fit(X, Y)
		predictions = clf.predict_proba(unknown)
		print('predictions for unknown data...')
		write_prediction(unknown_data_file + '_predictions_%s.csv.gz' % name, unknown_object_ids, predictions)
		print('predictions done after %.1fs' % (time() - t0))
	return clf


if os.environ.get('VIZ', '0') == '1':
	from yellowbrick.features import Rank1D, Rank2D, RadViz, ParallelCoordinates
	import matplotlib.pyplot as plt
	print("Rank1D...")
	features = train_columns
	visualizer = Rank1D(features=features, algorithm='shapiro')
	visualizer.fit(X, Y)                # Fit the data to the visualizer
	visualizer.transform(X)             # Transform the data
	visualizer.poof(outpath='viz_feature_rank1d.pdf', bbox_inches='tight')
	plt.close('all')
	feature_diversity = visualizer.ranks_
	# Instantiate the visualizer with the Covariance ranking algorithm
	print("Rank2D...")
	visualizer = Rank2D(features=features, algorithm='spearman')
	visualizer.fit(X, Y)                # Fit the data to the visualizer
	visualizer.transform(X)             # Transform the data
	visualizer.poof(outpath='viz_feature_rank2d.pdf', bbox_inches='tight')
	plt.close('all')
	
	"""
	# reorder the features so similar ones are together
	features_to_handle = list(range(len(features)))
	print(features_to_handle)
	features_ordered = []
	last_feature = 0
	numpy.random.seed(1)
	while features_to_handle:
		print("%d ..." % last_feature, feature_distance.shape)
		invdists = 1. / (visualizer.ranks_[last_feature,features_to_handle]**2 + 1e-5)
		invdists /= invdists.sum()
		i = numpy.random.choice(range(len(features_to_handle)), p=invdists)
		a = features_to_handle.pop(i)
		features_ordered.append(a)
		last_feature = a
	print(features_ordered)
	print(len(features))
	features = [train_columns[i] for i in features_ordered]
	print(features)
	print(len(features))
	XT = X[:,numpy.asarray(features_ordered)]
	print(X.shape, XT.shape, Y.shape)
	"""
	numpy.random.seed(1)
	XT = numpy.arange(1000).reshape((-1,2))
	YT = numpy.random.randint(6, size=len(XT))*10
	print(XT.shape, XT.dtype, YT.shape, YT.dtype)
	
	visualizer = RadViz(classes = [0,10,20,30,40,50])
	visualizer.fit_transform(XT, YT)
	visualizer.poof(outpath='viz_feature_radviz.pdf', bbox_inches='tight')
	plt.close('all')
	
	visualizer = ParallelCoordinates(classes = [0,10,20,30,40,50],
		sample=0.05, shuffle=True, fast=True)
	visualizer.fit_transform(XT, YT)
	visualizer.poof(outpath='viz_feature_parallel_coords.pdf', bbox_inches='tight')
	plt.close('all')
	XT = X
	YT = Y
	
	print(XT.shape, XT.dtype, YT.shape, YT.dtype)
	visualizer = RadViz(classes=labels)
	visualizer.fit_transform(XT, YT)
	visualizer.poof(outpath='viz_feature_radviz.pdf', bbox_inches='tight')
	plt.close('all')
	
	
	visualizer = ParallelCoordinates(classes=labels, sample=0.05, shuffle=True, fast=True)
	visualizer.fit_transform(XT, YT)
	visualizer.poof(outpath='viz_feature_parallel_coords.pdf', bbox_inches='tight')
	plt.close('all')
	sys.exit(0)
	clf = RandomForestClassifier(n_estimators=400, class_weight=class_weights)

if os.environ.get('FIND_FEATURE_SUBSET', '0') == '1':
	clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights)
	from sklearn.feature_selection import RFE, RFECV
	from sklearn.model_selection import StratifiedKFold
	print("Recursive feature elimination with RandomForest100...")
	t0 = time()
	rfe = RFE(estimator=clf, n_features_to_select=40, step=0.1, verbose=2)
	#rfe = RFECV(estimator=clf, min_features_to_select=10, step=0.1, verbose=2, 
	#	cv=StratifiedKFold(2), scoring=scorer)
	rfe.fit(X, Y)
	print('done after %.1fs' % (time() - t0))
	#print(rfe.grid_scores_)

	indices = numpy.argsort(rfe.ranking_)
	print("Feature ranking:")
	fcols = open('important_columns1.txt', 'w')
	for f, index in enumerate(indices):
		print("%d. feature %d (%d) -- %s" % (f + 1, index, rfe.ranking_[index], train_columns[index]))
		if rfe.ranking_[index] == 1:
			fcols.write("%d\t%s\n" % (index,train_columns[index]))
	
	premask = rfe.ranking_ <= 4
	#premask = rfe.ranking_ > 0
	Z = mytransformer.fit_transform(X[:,premask])
	print("Recursive feature elimination with SVC...")
	clf = LinearSVC(max_iter=10000, class_weight=class_weights)
	t0 = time()
	rfe = RFE(estimator=clf, n_features_to_select=30, step=0.1, verbose=2)
	#rfe = RFECV(estimator=clf, min_features_to_select=10, step=0.1, verbose=2, 
	#	cv=StratifiedKFold(2), scoring=scorer)
	rfe.fit(Z, Y)
	print('done after %.1fs' % (time() - t0))
	#print(rfe.grid_scores_)

	indices = numpy.argsort(rfe.ranking_)
	print("Feature ranking:")
	fcols = open('important_columns2.txt', 'w')
	for f, index in enumerate(indices):
		oldindex = numpy.where(premask)[0][index]
		print("%d. feature %d (%d) -- %s" % (f + 1, index, rfe.ranking_[index], train_columns[oldindex]))
		if rfe.ranking_[index] == 1:
			fcols.write("%d\t%s\n" % (index, train_columns[oldindex]))
	
	sys.exit(0)

#train_and_evaluate('KNN2', clf = KNeighborsClassifier(n_neighbors=2))
#train_and_evaluate('KNN4', clf = KNeighborsClassifier(n_neighbors=4))
#train_and_evaluate('KNN10', clf = KNeighborsClassifier(n_neighbors=10))
#train_and_evaluate('KNN40', clf = KNeighborsClassifier(n_neighbors=40))

#train_and_evaluate('XGradientBoosting', clf = XGBClassifier(n_estimators=40))
train_and_evaluate('RandomForest4', clf = RandomForestClassifier(n_estimators=4, class_weight=class_weights))
train_and_evaluate('RandomForest10', clf = RandomForestClassifier(n_estimators=10, class_weight=class_weights))
train_and_evaluate('RandomForest40', clf = RandomForestClassifier(n_estimators=40, class_weight=class_weights))
train_and_evaluate('RandomForest100', clf = RandomForestClassifier(n_estimators=100, class_weight=class_weights))
train_and_evaluate('RandomForest400', clf = RandomForestClassifier(n_estimators=400, class_weight=class_weights))
train_and_evaluate('AdaBoost40', clf = AdaBoostClassifier(n_estimators=40))
train_and_evaluate('AdaBoost400', clf = AdaBoostClassifier(n_estimators=400))
train_and_evaluate('ExtraTrees40', clf = ExtraTreesClassifier(n_estimators=40))
#train_and_evaluate('RandomForest4000', clf = RandomForestClassifier(n_estimators=4000))

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


