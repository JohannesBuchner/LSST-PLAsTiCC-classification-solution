from __future__ import print_function, division
from alltrain import *
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
X_galmask = train.hostgal_photoz.values == 0
# remove some columns that we do not want to use
#X = X[:,5:] # RA, DEC, l, b, ddf

print('train gal/exgal:', X_galmask.sum(), (~X_galmask).sum())
#qt = QuantileTransformer(output_distribution='normal')
qt = MinMaxScaler(feature_range=(-1,1))
X = qt.fit_transform(X)
#X = X[:,5:] # specz, photoz, photozerr, distmod, mwebv

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id').values
	unknown_galmask = unknown.hostgal_photoz.values == 0
	print('test gal/exgal:', unknown_galmask.sum(), (~unknown_galmask).sum())
	unknown = unknown.values
	unknown = unknown[:,5:] # remove columns RA, DEC, l, b, ddf
	unknown = unknown[:,5:] # specz, photoz, photozerr, distmod, mwebv
	print('unknown:', unknown.shape)
	unknown = imp.transform(unknown)
	unknown = qt.transform(unknown)
	if simplify_space:
		unknown = unknown[:,column_mask]

classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]

def reassign_mapping(name, X, Y, Z, pca, clf):
	print()
	print("running %s ..." % name)
	print('input shapes:', X.shape, Y.shape, Z.shape)
	t0 = time()
	XZ = pca.fit_transform(numpy.vstack((X, Z)))
	print("PCA dimensionality reduction done after %0.3fs" % (time() - t0))
	del Z, X
	print('PCA Variance ratios:', pca.explained_variance_ratio_)

	t0 = time()
	clusters_all = clf.fit_predict(XZ)
	print("clustering done after %0.3fs" % (time() - t0))
	del XZ
	clusters = clusters_all[:len(Y)]
	newclusters = clusters_all[len(Y):]
	clusterids = numpy.unique(clusters_all)
	del clusters_all
	Ylist = set(numpy.unique(Y))
	N0 = numpy.array([1 if cls in Y else 0 for cls in classes], dtype='f')

	# go through each cluster
	# compute fraction in training sample assigned to that cluster
	# check if very few -> add 99
	result = numpy.zeros((len(newclusters), len(classes)))
	for cluster in clusterids:
		mask_train = clusters == cluster
		N1 = numpy.array([(Y[mask_train] == cls).sum() for cls in classes], dtype='f')
		N = N1 + N0 * FLATPRIOR_STRENGTH
		N[-1] = OUTLIERS_STRENGTH
		mask_test = newclusters == cluster
		prob = N / N.sum()
		prob = numpy.where(prob == 0, 0, prob**PROB_FLATNESS)
		prob /= prob.sum()
		#Nstr = ' '.join(['%3d' % Ni for Ni in N])
		Nstr = ' '.join(['%2d' % (pi*100) for pi in prob])
		print('cluster %2d:%4d/%6d | %s %s' % (cluster, mask_train.sum(), mask_test.sum(), Nstr, "***" if N1.sum() < 2 else ""))
		result[mask_test,:] = prob / prob.sum()
	return result

"""

This technique applies kmeans clustering to the test and training data, 
partitioning it into groups.
From the training data samples in each group, the class fractions are inferred.
One or two outliers are assumed in each group as well.

This forms a simple classifier with few tuning parameters that automatically
incorporates novelty detection.

It is similar to NearestCentroid classification, but probabilistic.

"""

# parameters
# flattening of predictor (to even out imbalanced classes) recommended values: 1 - 0.2
PROB_FLATNESS = float(os.environ.get('PROB_FLATNESS', '0.2'))
# prior of relevant classes (to compensate for a missing sample) recommended values: ?
FLATPRIOR_STRENGTH = float(os.environ.get('FLATPRIOR_STRENGTH', '0.1'))
# how many outliers to expect (they are not in the training sample) recommended values: ?
OUTLIERS_STRENGTH = float(os.environ.get('OUTLIERS_STRENGTH', '2.0'))

k = int(os.environ.get('K', '20'))
n_components = int(os.environ.get('NPCACOMP', '40'))

prefix = ('SIMPLE' if simplify_space else '') + 'PCA%d' % n_components
pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)

for name in [prefix + '-Kmeans%d' % k]:#, prefix + '-HierCluster%d' % k]:
	if 'Kmeans' in name:
		clf = KMeans(n_clusters=k)
	elif 'HierCluster' in name:
		clf = AgglomerativeClustering(n_clusters=k, affinity='cosine', linkage='average')
	else:
		assert False

	pred = numpy.empty((len(unknown_galmask), len(classes)))
	print("making exgal predictions...")
	pred[~unknown_galmask,:] = reassign_mapping(name, X[~X_galmask,:], Y[~X_galmask], unknown[~unknown_galmask,:], pca=pca, clf=clf)
	#pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
	print("making gal predictions...")
	pred[unknown_galmask,:] = reassign_mapping(name, X[X_galmask,:], Y[X_galmask], unknown[unknown_galmask,:], pca=pca, clf=clf)
	filename = 'prediction_out_%s_flatness%s_prior%s_outliers%s.csv.gz' % (name, PROB_FLATNESS, FLATPRIOR_STRENGTH, OUTLIERS_STRENGTH)
	print("storing under '%s' ..." % filename)
	numpy.savetxt(filename,
		numpy.hstack((unknown_object_ids.reshape((-1,1)), pred)), 
		fmt='%d' + ',%.3e' * len(classes),
		delimiter=',',
		header='object_id,class_6,class_15,class_16,class_42,class_52,class_53,class_62,class_64,class_65,class_67,class_88,class_90,class_92,class_95,class_99',
		comments='')
	



