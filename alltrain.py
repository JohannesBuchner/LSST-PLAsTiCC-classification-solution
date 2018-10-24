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
weights_targets = numpy.zeros(100)
for l, N in zip(labels, N_labels):
	weights_targets[l] = 1. / (N + 0.1)
weights_targets[99] *= 2
weights_targets[64] *= 2
weights_targets[15] *= 2
custom_class_weights = numpy.array([weights_targets[l] for l in labels])

Y = Y_orig
X = train.values

simplify_space = os.environ.get('SIMPLIFY', '0') == '1'
if simplify_space:
	important_columns = set([line.split()[1] for line in open('important_columns.txt')])
	column_mask = numpy.array([c in important_columns for c in train.columns if c not in ('object_id', 'target')])
	X = X[:,column_mask]
else:
	column_mask = numpy.array([True for c in train.columns])

#X[~numpy.isfinite(X)] = -99
imp = SimpleImputer(missing_values=numpy.nan, strategy='mean')
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



