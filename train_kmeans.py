from __future__ import print_function, division
from alltrain import *
from sklearn.cluster import KMeans, MiniBatchKMeans
X_galmask = train.hostgal_photoz.values == 0
qt = QuantileTransformer()
X = qt.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')
	unknown_galmask = unknown.hostgal_photoz.values == 0
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	unknown = qt.transform(unknown)

classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]

def reassign_mapping(clusters, labels, newclusters):
	# go through each cluster
	# compute fraction in training sample assigned to that cluster
	# check if very few -> add 99
	clusters = clusters.argmin(axis=1)
	newclusters = newclusters.argmin(axis=1)
	result = numpy.zeros((len(newclusters), len(classes)))
	print('mapping input:', clusters.shape, labels.shape, newclusters.shape)
	print(clusters)
	for cluster in numpy.unique(clusters):
		mask_train = clusters == cluster
		N = numpy.array([(labels[mask_train] == cls).sum() for cls in classes], dtype='f')
		N[-1] = 2.
		mask_test = newclusters == cluster
		prob = N / N.sum()
		prob = numpy.where(prob == 0, 0, prob**0.04)
		Nstr = ' '.join(['%3d' % Ni for Ni in N])
		Nstr = ' '.join(['%2d' % (pi*100) for pi in prob])
		print('cluster %2d: %5d/%5d | %s' % (cluster, mask_train.sum(), mask_test.sum(), Nstr))
		result[mask_test,:] = prob / prob.sum()
	return result

k = int(os.environ.get('K', '8'))
name = 'Kmeans%d' % k
clf = KMeans(n_clusters=k)

t0 = time()
sys.stdout.write('running %s ...\r' % name)
sys.stdout.flush()
unknown_clusters = clf.fit_transform(unknown[unknown_galmask,:])
known_clusters = clf.transform(X[X_galmask,:])
print("done in %0.3fs" % (time() - t0))
print('gal:', numpy.shape(unknown_clusters), numpy.shape(known_clusters), Y[X_galmask].shape)
unknown_pred = reassign_mapping(known_clusters, Y[X_galmask], unknown_clusters)

t0 = time()
sys.stdout.write('running %s ...\r' % name)
sys.stdout.flush()
unknown_clusters = clf.fit_transform(unknown[~unknown_galmask,:])
known_clusters = clf.transform(X[~X_galmask,:])
print("done in %0.3fs" % (time() - t0))
print('exgal:', numpy.shape(unknown_clusters), numpy.shape(known_clusters), Y[X_galmask].shape)
unknown_pred = reassign_mapping(known_clusters, Y[~X_galmask], unknown_clusters)
print("done.")



