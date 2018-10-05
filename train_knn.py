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
import sys, os
import pandas

training_data_file = 'training_set_all.csv.gz'
unknown_data_file = os.environ.get('PREDICT_FILE')


encoder = LabelBinarizer()
train = pandas.read_csv(training_data_file)

train.pop('object_id')
Y_orig = train.pop('target').values
encoder.fit(Y_orig)
labels = encoder.classes_
#N_labels, _ = numpy.histogram(Y_orig, bins=labels)
N_labels = numpy.array([(Y_orig == l).sum() for l in labels])
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
weights_targets = numpy.zeros(100)
for l, N in zip(labels, N_labels):
	weights_targets[l] = 1./N
weights_targets[99] *= 2
weights_targets[64] *= 2
weights_targets[15] *= 2

#weights_labels = numpy.ones(len(labels))
#weights_labels[labels == 99] = 2
#weights_labels[labels == 64] = 2
#weights_labels[labels == 15] = 2

#Y = encoder.fit_transform(Y_orig)
Y = Y_orig
X = train.values
print('data:', X.shape, Y.shape)
X[~numpy.isfinite(X)] = -99
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

# not used atm
def my_log_loss(y_true, y_pred, eps=1e-15, normalize=True, labels=None):
	transformed_labels = encoder.transform(y_true)
	y_pred = numpy.clip(y_pred, eps, 1 - eps)
	y_pred /= y_pred.sum(axis=1)[:, numpy.newaxis]
	
	sample_weight = weights_targets[y_true]
	loss = (transformed_labels * numpy.log(y_pred)).sum(axis=1)
	return _weighted_sum(loss, sample_weight, normalize)
	#return -(loss * weighting).sum() / weighting.sum()

scorer = make_scorer(my_log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)

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
		numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',')
		clf.fit(X_white, Y)
		predictions = clf.predict_proba(unknown_white)
		print('predictions for unknown data...')
		numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, predictions, delimiter=',')
		print('predictions done after %.1fs' % (time() - t0))
	return clf

for n_components in 40,:
	print("dimensionality reduction with PCA-%d" % n_components)
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X)
	print("done in %0.3fs" % (time() - t0))
	X_white = pca.transform(X)
	if execute:
		unknown_white = pca.transform(unknown)

	train_and_evaluate('KNN2', clf = KNeighborsClassifier(n_neighbors=2))
	train_and_evaluate('KNN4', clf = KNeighborsClassifier(n_neighbors=4))
	train_and_evaluate('KNN8', clf = KNeighborsClassifier(n_neighbors=8))
	#train_and_evaluate('KNN40', clf = KNeighborsClassifier(n_neighbors=40))
	#train_and_evaluate('KNN10-r', clf = KNeighborsClassifier(n_neighbors=10, weights='distance'))
	#train_and_evaluate('KNN40-r', clf = KNeighborsClassifier(n_neighbors=40, weights='distance'))

	train_and_evaluate('RandomForest4', clf = RandomForestClassifier(n_estimators=4))
	train_and_evaluate('RandomForest10', clf = RandomForestClassifier(n_estimators=10))
	train_and_evaluate('RandomForest40', clf = RandomForestClassifier(n_estimators=40))
	#train_and_evaluate('RandomForest100', clf = RandomForestClassifier(n_estimators=100))
	#train_and_evaluate('AdaBoost', clf = AdaBoostClassifier(n_estimators=40))
	#train_and_evaluate('GradientBoosting', clf = GradientBoostingClassifier(n_estimators=40))
	#train_and_evaluate('ExtraTrees', clf = ExtraTreesClassifier(n_estimators=40))
	
	train_and_evaluate('MLP10', clf = MLPClassifier(hidden_layer_sizes=(10,)))
	train_and_evaluate('MLP10-20-10', clf = MLPClassifier(hidden_layer_sizes=(10,20,10)))
	train_and_evaluate('MLP4-16-4', clf = MLPClassifier(hidden_layer_sizes=(4,16,4)))
	train_and_evaluate('MLP40', clf = MLPClassifier(hidden_layer_sizes=(40,)))
	
	break


