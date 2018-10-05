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

methods = 'KNN2', 'KNN4', 'RandomForest4'

A = []
B = []
Y = numpy.loadtxt('training_set_target.csv')

for method in methods:
	A.append(numpy.loadtxt(training_data_file + '_predictions_%s.csv.gz' % method, delimiter=','))
	
	B.append(numpy.loadtxt(unknown_data_file + '_predictions_%s.csv.gz' % method, delimiter=','))

X = numpy.hstack(tuple(A))
Z = numpy.hstack(tuple(B))

# we want to train something that predicts T from A


encoder = LabelBinarizer()
encoder.fit(Y)
labels = encoder.classes_
#N_labels, _ = numpy.histogram(Y_orig, bins=labels)
N_labels = numpy.array([(Y == l).sum() for l in labels])
weights_labels = numpy.ones(len(labels))
# https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
weights_labels[labels == 99] = 2
weights_labels[labels == 64] = 2
weights_labels[labels == 15] = 2

# not used atm
def my_log_loss(y_true, y_pred, eps=1e-15, normalize=True, labels=None):
	transformed_labels = encoder.transform(y_true)
	y_pred = numpy.clip(y_pred, eps, 1 - eps)
	y_pred /= y_pred.sum(axis=1)[:, numpy.newaxis]
	
	weighting = weights_labels / N_labels
	loss = (transformed_labels * numpy.log(y_pred)).sum(axis=0)
	return -(loss * weighting).sum() / weighting.sum()

scorer = make_scorer(log_loss, eps=1e-15, greater_is_better=False, needs_proba=True, labels=labels)

clf = MLPClassifier(hidden_layer_sizes=40)

print("training MLP...")
q = cross_val_score(clf, X, Y, cv=5, scoring=scorer)
print(q.mean(), q.std())

t0 = time()
print('predictions for training data...')
predictions = cross_val_predict(clf, X, Y, method='predict_proba')
numpy.savetxt(training_data_file + '_predictions.csv.gz' % name, predictions, delimiter=',')
clf.fit(X, Y)
predictions = clf.predict_proba(Z)
print('predictions for unknown data...')
numpy.savetxt(unknown_data_file + '_predictions.csv.gz' % name, predictions, delimiter=',')
print('predictions done after %.1fs' % (time() - t0))



