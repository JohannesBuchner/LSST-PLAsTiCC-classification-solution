from __future__ import print_function, division
from alltrain import *

train_columns = list(train.columns)
Y = train['hostgal_photoz'].values == 0
del train

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown_object_ids = unknown.pop('object_id').values
	unknownY = unknown['hostgal_photoz'].values == 0
	unknown = unknown.values
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)

ids = numpy.hstack((training_object_ids, unknown_object_ids))
Y = numpy.hstack((Y, unknownY))
print(X.shape, unknown.shape)
X = numpy.vstack((X, unknown))
print("combined:", X.shape)
name = 'RandomForest40'
clf = RandomForestClassifier(n_estimators=40, class_weight='balanced')

print()
sys.stdout.write('running %s ...\r' % name)
sys.stdout.flush()
t0 = time()
prediction = cross_val_predict(clf, X, Y, cv=4)
mask = prediction != Y
print('%2.3f%% wrong %s (training speed: %.1fs)' % (mask.mean()*100, name, time() - t0))

numpy.savetxt('galsplit_RF40.txt.gz', ids[mask], fmt='%d')


