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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.impute import SimpleImputer
#from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics.classification import _weighted_sum
import numpy
from numpy import log10, exp, pi, log
import sys, os
import pandas

training_data_file = os.environ['TRAINING_FILE'] #'training_set_all.csv.gz'
unknown_data_file = os.environ.get('PREDICT_FILE')

encoder = LabelBinarizer()
train = pandas.read_csv(training_data_file)

training_object_ids = train.pop('object_id')
Y_orig = train.pop('target').values
encoder.fit(Y_orig)
labels = encoder.classes_
#N_labels, _ = numpy.histogram(Y_orig, bins=labels)
N_labels = numpy.array([(Y_orig == l).sum() for l in labels])
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
# compute class weights inversely proportional to frequency
class_weights = {l: len(Y_orig) / (len(labels) * (Y_orig == l).sum()) for l in labels}
if 64 in class_weights:
	class_weights[64] = class_weights[64] * 2
if 15 in class_weights:
	class_weights[15] = class_weights[15] * 2
class_weights = 'balanced'
print("Class weights:", class_weights)
print("Class numbers:", list(zip(labels, N_labels)))
weights_targets = numpy.zeros(100)
for l, N in zip(labels, N_labels):
	weights_targets[l] = 1. / (N + 0.1)
weights_targets[99] *= 2
weights_targets[64] *= 2
weights_targets[15] *= 2
custom_class_weights = numpy.array([weights_targets[l] for l in labels])

Y = Y_orig
X = train.values

valid_columns = numpy.array([numpy.isfinite(X[:,i]).any() for i in range(X.shape[1])])
simplify_space = os.environ.get('SIMPLIFY', '0') == '1'
if simplify_space:
	important_columns = set([line.split()[1] for line in open('important_columns.txt')])
	blacklist_columns = set([line.strip() for line in open('blacklist_features.txt')])
	column_mask = numpy.array([c in important_columns and c not in blacklist_columns
		for c in train.columns if c not in ('object_id', 'target')])
	X = X[:,column_mask]
	assert len(train.columns[column_mask]) == X.shape[1]
else:
	column_mask = numpy.array([True for c in train.columns])

# imputing also removes columns which have no useful values, so we need to know those
#imp = SimpleImputer(missing_values=numpy.nan, strategy='mean', copy=False)
#imp = SimpleImputer(missing_values=numpy.nan, strategy='constant', fill_value=-99, copy=False)
impute = os.environ.get('IMPUTE', '1') == '1'
if impute:
	valid_column_mask = numpy.logical_and(column_mask, valid_columns)
	imp = SimpleImputer(missing_values=numpy.nan, strategy='median', copy=False)
else:
	valid_column_mask = column_mask
	class NoImpute(object):
		def fit_transform(self, X): return X
		def transform(self, X): return X
	imp = NoImpute()

#X[~numpy.isfinite(X)] = -99
X = imp.fit_transform(X)

transform = os.environ.get('TRANSFORM', 'MM')
if transform == 'QTU':
	mytransformer = QuantileTransformer()
elif transform == 'QTN':
	mytransformer = QuantileTransformer(output_distribution='normal')
elif transform == 'MM':
	mytransformer = MinMaxScaler(feature_range=(-1,1))
elif transform == 'NORM':
	mytransformer = StandardScaler()
else:
	assert False, ('unknown transform requested:', transform)

def write_prediction(filename, object_ids, proba, outlierproba=None):
	i = 0
	all_classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
	df = pandas.DataFrame()
	df['object_id'] = object_ids
	for j, cls in enumerate(all_classes):
		if cls in labels:
			v = proba[:,i]
			i += 1
		else:
			v = numpy.zeros(len(proba))
		df['class_%d' % cls] = v
	if outlierproba is not None:
		df['class_99'] = outlierproba
	
	#header = "object_id," + ','.join(['class_%d' % cls for cls in all_classes])
	#numpy.savetxt(filename, proba_all, delimiter=',', fmt='%d' + ',%.4e'*15, 
	#	header=header, comments='')
	df.to_csv(filename, index=False, float_format='%.4e', compression='gzip')

def my_log_loss(y_true, y_pred, eps=1e-15, normalize=True, labels=None):
	transformed_labels = encoder.transform(y_true)
	y_pred = numpy.clip(y_pred, eps, 1 - eps)
	y_pred /= y_pred.sum(axis=1)[:, numpy.newaxis]
	
	sample_weight = weights_targets[y_true]
	loss = (transformed_labels * numpy.log(y_pred)).sum(axis=1)
	return _weighted_sum(loss, sample_weight, normalize)
	#return -(loss * weighting).sum() / weighting.sum()

from logloss import plasticc_log_loss

scorer = make_scorer(plasticc_log_loss, eps=1e-15, custom_class_weights=custom_class_weights, greater_is_better=False, needs_proba=True, labels=labels)

scorer = make_scorer(my_log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)



