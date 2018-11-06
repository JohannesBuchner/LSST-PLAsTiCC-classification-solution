from __future__ import print_function, division
from time import time
from sklearn import preprocessing
from sklearn.preprocessing import quantile_transform, QuantileTransformer, MinMaxScaler, StandardScaler
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
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
#from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics.classification import _weighted_sum
import numpy
import sys, os
import pandas

from sklearn.externals import joblib

print("loading training data...")
training_data_file = 'training_set_SED_features.txt'
train = pandas.read_csv(training_data_file)

encoder = LabelBinarizer()
train = pandas.read_csv(training_data_file)
Y = train.pop('target').values
sample_weights = train.pop('weight').values
encoder.fit(Y)
mytransformer = MinMaxScaler(feature_range=(-1,1))
X = train.values
del train
X[X<-4] = -4
X = mytransformer.fit_transform(X)
labels = encoder.classes_

N_labels = numpy.array([(Y == l).sum() for l in labels])
weights_targets = numpy.zeros(100)
for l, N in zip(labels, N_labels):
	weights_targets[l] = 1. / (N + 0.1)
weights_targets[99] *= 2
weights_targets[64] *= 2
weights_targets[15] *= 2

from logloss import my_log_loss
scorer = make_scorer(my_log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels, encoder=encoder, weights_targets=weights_targets)
#scorer = make_scorer(log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)


def train_and_evaluate(name, clf, **kwargs):
	print('running %s ...' % name)
	t0 = time()
	q = cross_val_score(clf, XT, Y, cv=4, scoring=scorer, n_jobs=4, **kwargs)
	print('%2.2f +- %2.2f %s (training speed: %.1fs)' % (q.mean(), q.std(), name, time() - t0))

def get_SED_transformer():
	try:
		return joblib.load('SEDclassifier-MLP.joblib')
	except IOError:
		t0 = time()
		print('Training MLP4 ...')
		#clf = LinearDiscriminantAnalysis()
		clf = MLPClassifier(hidden_layer_sizes=10, activation='tanh', max_iter=2000)
		clf.fit(X, Y)
		print('Trained MLP4 in %.1fs' % (time() - t0))
		joblib.dump(clf, 'SEDclassifier-MLP.joblib')
		return clf

if __name__ == '__main__':
	#train_and_evaluate('KNN100', clf = KNeighborsClassifier(n_neighbors=100))
	for whiten_colors, whiten_simplify in (False, False), (True, False), (True,True):
		print()
		XT = X.copy()
		if whiten_colors:
			t0 = time()
			pca = PCA(n_components=6, svd_solver='randomized', whiten=True)
			print("Color whitening PCA done in %0.3fs" % (time() - t0))
			XT[:,:6] = pca.fit_transform(XT[:,:6])
		if whiten_simplify:
			t0 = time()
			pca = PCA(n_components=5, svd_solver='randomized', whiten=True)
			print("PCA-5 done in %0.3fs" % (time() - t0))
			XT = pca.fit_transform(XT)
		
		for activation in 'tanh','relu':
			for n in 1, 2, 3, 4, 10, 40, (10,20,10):
				train_and_evaluate('MLP%s' % str(n), 
					clf = MLPClassifier(hidden_layer_sizes=n, 
						activation=activation, max_iter=2000))
		
		train_and_evaluate('LDA', clf = LinearDiscriminantAnalysis())
		
		train_and_evaluate('RandomForest400', clf = RandomForestClassifier(n_estimators=400, class_weight='balanced'), fit_params=dict(sample_weight=sample_weights))
		# does not finish:
		#if whiten_simplify:
		#	train_and_evaluate('SVC', clf = SVC(probability=True, class_weight='balanced'), fit_params=dict(sample_weight=sample_weights))



