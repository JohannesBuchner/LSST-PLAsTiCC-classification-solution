from __future__ import print_function, division
from alltrain import *
del train
import matplotlib.pyplot as plt

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

X = mytransformer.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	print('reading data file to predict ...')
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id').values
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)
	unknown = mytransformer.transform(unknown)

def isolate_with(name, clf):
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	clf.fit(X)
	predictions = clf.predict(X)
	print('running %s: training speed: %.1fs' % (name, time() - t0))
	plt.hist(predictions, histtype='step', label=name)
	plt.savefig('train_novel_%s.pdf' % name, bbox_inches='tight')
	
	if execute:
		print('predictions for unknown data...')
		t0 = time()
		predictions = clf.predict(unknown)
		i = numpy.where(predictions == -1)[0]
		print('novel: %d/%d (%.2f%%)' % (len(i), len(unknown), len(i) * 100. / len(unknown)))
		numpy.savetxt(unknown_data_file + '_novel_%s.csv' % name, numpy.transpose([i, unknown_object_ids[i]]), delimiter=',', fmt='%d')
		print('predictions done after %.1fs' % (time() - t0))


for outlier_fraction in 0.4, 0.1, 0.04, 0.01, 0.001:
	prefix = ('SIMPLE' if simplify_space else '') + transform + '-'
	isolate_with(prefix + 'EllEnvelope-%s' % outlier_fraction,
		EllipticEnvelope(contamination=outlier_fraction))
	isolate_with(prefix + 'IsolForest-%s' % outlier_fraction, 
		IsolationForest(n_estimators=100, contamination=outlier_fraction, behaviour='new'))
	





