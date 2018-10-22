import numpy

def weight_sum(per_class_metrics, weight_vector, norm=True):
	"""
	Calculates the weighted metric
	Parameters
	----------
	per_class_metrics: numpy.float
	the scores separated by class (a list of arrays)
	weight_vector: numpy.ndarray floar
	The array of weights per class
	norm: boolean, optional
	Returns
	-------
	weight_sum: numpy.float
	The weighted metric
	"""
	weight_sum = numpy.dot(weight_vector, per_class_metrics)

	return weight_sum

def check_weights(M, avg_info, chosen=None, truth=None):
	"""
	Converts standard weighting schemes to weight vectors for weight_sum
	Parameters
	----------
	avg_info: str or numpy.ndarray, float
	keyword about how to calculate weighted average metric
	M: int
	number of classes
	chosen: int, optional
	which class is to be singled out for down/up-weighting
	truth: numpy.ndarray, int, optional
	true class assignments
	Returns
	-------
	weights: numpy.ndarray, float
	relative weights per class
	Notes
	-----
	Assumes a random class
	"""
	if type(avg_info) != str:
		avg_info = numpy.asarray(avg_info)
		weights = avg_info / numpy.sum(avg_info)
		assert(numpy.isclose(sum(weights), 1.))
	elif avg_info == 'per_class':
		weights = numpy.ones(M) / float(M)
	elif avg_info == 'per_item':
		classes, counts = numpy.unique(truth, return_counts=True)
		weights = numpy.zeros(M)
		weights[classes] = counts / float(len(truth))
		assert len(weights) == M
	elif avg_info == 'flat':
		weights = numpy.ones(M)
	elif avg_info == 'up' or avg_info == 'down':
		if chosen is None:
			chosen = numpy.random.randint(M)
		if avg_info == 'up':
			weights = numpy.ones(M) / numpy.float(M)
			weights[chosen] = 1.
		elif avg_info == 'down':
			weights = numpy.ones(M)
			weights[chosen] = 1./numpy.float(M)
		else:
			print('something has gone wrong with avg_info '+str(avg_info))
	return weights


def det_to_prob(dets, prediction=None):
	"""
	Reformats vector of class assignments into matrix with 1 at true/assigned class and zero elsewhere
	Parameters
	----------
	dets: numpy.ndarray, int
	vector of classes
	prediction: numpy.ndarray, float, optional
	predicted class probabilities
	Returns
	-------
	probs: numpy.ndarray, float
	matrix with 1 at input classes and 0 elsewhere
	Notes
	-----
	formerly truth_reformatter
	Does not yet handle number of classes in truth not matching number of classes in prediction, i.e. for having "other" class or secret classes not in training set.  The prediction keyword is a kludge to enable this but should be replaced.
	"""
	N = len(dets)
	indices = range(N)

	if prediction is None:
		prediction_shape = (N, int(numpy.max(dets) + 1))
	else:
		prediction, dets = numpy.asarray(prediction), numpy.asarray(dets)
		prediction_shape = numpy.shape(prediction)

	probs = numpy.zeros(prediction_shape)
	probs[indices, dets] = 1.

	return probs

truth_reformatter = det_to_prob


def sanitize_predictions(predictions, epsilon=1.e-8):
	"""
	Replaces 0 and 1 with 0+epsilon, 1-epsilon
	Parameters
	----------
	predictions: numpy.ndarray, float
	N*M matrix of probabilities per object, may have 0 or 1 values
	epsilon: float
	small placeholder number, defaults to floating point precision
	Returns
	-------
	predictions: numpy.ndarray, float
	N*M matrix of probabilities per object, no 0 or 1 values
	"""
	assert epsilon > 0. and epsilon < 0.0005
	mask1 = (predictions < epsilon)
	mask2 = (predictions > 1.0 - epsilon)

	predictions[mask1] = epsilon
	predictions[mask2] = 1.0 - epsilon
	predictions = predictions / numpy.sum(predictions, axis=1)[:, numpy.newaxis]
	return predictions


def averager(per_object_metrics, truth, M, vb=False):
	"""
	Creates a list with the metrics per object, separated by class
	Notes
	-----
	There is currently a kludge for when there are no true class members, causing an improvement when that class is upweighted due to increasing the weight of 0.
	"""
	group_metric = per_object_metrics
	class_metric = numpy.empty(M)
	for m in range(M):
		true_indices = numpy.where(truth == m)[0]
		how_many_in_class = len(true_indices)
		try:
			assert(how_many_in_class > 0)
			per_class_metric = group_metric[true_indices]
			# assert(~numpy.all(numpy.isnan(per_class_metric)))
			class_metric[m] = numpy.average(per_class_metric)
		except AssertionError:
			class_metric[m] = 0.
		if vb: print('by request '+str((m, how_many_in_class, class_metric[m])))
	return class_metric

def plasticc_log_loss(truth, prediction, custom_class_weights=1., eps=1e-15, normalize=True, labels=None):
	prediction, truth = numpy.asarray(prediction), numpy.asarray(truth)
	prediction_shape = numpy.shape(prediction)
	(N, M) = prediction_shape

	weights = check_weights(M, avg_info = 'per_class', truth=truth) * custom_class_weights
	print('weights:', weights.shape, weights)
	truth_mask = truth_reformatter(truth, prediction)

	prediction = sanitize_predictions(prediction, epsilon=eps)

	log_prob = numpy.log(prediction)
	logloss_each = -1. * numpy.sum(truth_mask * log_prob, axis=1)[:, numpy.newaxis]
	print('logloss_each:', logloss_each.shape)

	# use a better structure for checking keyword support
	class_logloss = averager(logloss_each, truth, M)

	logloss = weight_sum(class_logloss, weight_vector=weights)

	assert(~numpy.isnan(logloss))
	print('logloss eval:', logloss)

	return logloss


if __name__ == '__main__':
	K = 12
	N = 1000
	truth = numpy.random.randint(0, K, size=N)
	pred = numpy.ones((N, K)) / K
	print(plasticc_log_loss(truth, pred))
	

