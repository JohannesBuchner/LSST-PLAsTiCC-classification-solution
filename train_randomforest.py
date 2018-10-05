from time import time
from sklearn import preprocessing
from sklearn.preprocessing import quantile_transform, QuantileTransformer
from sklearn.decomposition import PCA, KernelPCA
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, make_scorer
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import RidgeClassifier, RidgeClassifierCV, LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.classification import _weighted_sum
import numpy
import sys
import pandas

training_data_file = 'training_set_all.csv.gz'
unknown_data_file = 'test_set_part0.csv.gz'


encoder = LabelBinarizer()
train = pandas.read_csv(training_data_file)

train.pop('object_id')
Y_orig = train.pop('target').values
encoder.fit(Y_orig)
labels = encoder.classes_
#N_labels, _ = numpy.histogram(Y_orig, bins=labels)
N_labels = numpy.array([(Y_orig == l).sum() for l in labels])
weights_labels = numpy.ones(len(labels))
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
weights_labels[labels == 99] = 2
weights_labels[labels == 64] = 2
weights_labels[labels == 15] = 2

#Y = encoder.fit_transform(Y_orig)
Y = Y_orig
X = train.values
print('data:', X.shape, Y.shape)
X[~numpy.isfinite(X)] = -99
qt = QuantileTransformer()
X = qt.fit_transform(X)

execute = True
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')        
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	unknown = qt.transform(unknown)

# not used atm
def my_log_loss(y_true, y_pred, eps=1e-15, normalize=True, labels=None):
	transformed_labels = encoder.transform(y_true)
	y_pred = numpy.clip(y_pred, eps, 1 - eps)
	y_pred /= y_pred.sum(axis=1)[:, numpy.newaxis]
	
	weighting = weights_labels / N_labels
	loss = (transformed_labels * numpy.log(y_pred)).sum(axis=0)
	return -(loss * weighting).sum() / weighting.sum()

scorer = make_scorer(log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)

def train_and_evaluate(name, clf):
	print()
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	q = cross_val_score(clf, X, Y, cv=5, scoring=scorer)
	print('%2.2f +- %2.2f %s (training speed: %.1fs)' % (q.mean(), q.std(), name, time() - t0))

	if not execute:
		# manual masking of one third
		X_train, X_test, y_train, y_test = train_test_split(X_white, Y, test_size = 0.25, random_state = 21)
		clf = clf.fit(X_train, y_train)
		t0 = time()
		for i in range(10):
			y_pred = clf.predict(X_test)
		
		#y_scores = clf.predict_proba(X_ts)
		if hasattr(clf, 'feature_importances_'):
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
		predictions = cross_val_predict(clf, X, Y, method='predict_proba')
		numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',')
		clf.fit(X, Y)
		predictions = clf.predict_proba(unknown)
		print('predictions for unknown data...')
		numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',')
		print('predictions done after %.1fs' % (time() - t0))
	return clf


train_and_evaluate('RandomForest4', clf = RandomForestClassifier(n_estimators=4))
train_and_evaluate('RandomForest10', clf = RandomForestClassifier(n_estimators=10))
train_and_evaluate('RandomForest40', clf = RandomForestClassifier(n_estimators=40))
train_and_evaluate('AdaBoost', clf = AdaBoostClassifier(n_estimators=40))
## too slow
# train_and_evaluate('GradientBoosting', clf = GradientBoostingClassifier(n_estimators=10)) # extremely slow
train_and_evaluate('ExtraTrees', clf = ExtraTreesClassifier(n_estimators=40))
## can't predict probabilities
#train_and_evaluate('RidgeClassifier', clf = RidgeClassifier())
#train_and_evaluate('SVC-default', clf = SVC(probability=True))
#train_and_evaluate('SVC-0.1', clf = SVC(probability=True, C = 0.1, gamma = 0.05))
#train_and_evaluate('LinearSVC-default', clf = LinearSVC(probability=True))
#train_and_evaluate('LinearSVC-0.1', clf = LinearSVC(probability=True, C = 0.1, gamma = 0.05))


