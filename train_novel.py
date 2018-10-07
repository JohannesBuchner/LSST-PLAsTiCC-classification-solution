from __future__ import print_function, division
from alltrain import *
import matplotlib.pyplot as plt

qt = QuantileTransformer()
X = qt.fit_transform(X)

name = 'IsolationForest'
clf = IsolationForest(n_estimators=400, contamination=0.01, behaviour='new')

sys.stdout.write('running %s ...\r' % name)
sys.stdout.flush()
t0 = time()
clf.fit(X)
predictions = clf.predict(X)
print('running %s: training speed: %.1fs' % (name, time() - t0))
plt.hist(predictions)
plt.savefig('train_novel.pdf', bbox_inches='tight')
plt.close()
execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id')        
	unknown = unknown.values                  
	print('unknown:', unknown.shape)
	unknown[~numpy.isfinite(unknown)] = -99
	unknown = qt.transform(unknown)

print('predictions for unknown data...')
t0 = time()
predictions = clf.predict(unknown)
i, _ = numpy.where(predictions == -1)
print('novel: %d/%d (%.2f%%)' % (len(i), len(unknown), len(i) * 100. / len(unknown)), unknown_object_ids[i])
numpy.savetxt(unknown_data_file + '_novel.csv.gz', predictions, delimiter=',')
print('predictions done after %.1fs' % (time() - t0))




