from collections import defaultdict
import numpy
import hashlib
import pandas

bands = range(6)

a = pandas.read_csv('training_set.csv')
b = pandas.read_csv('training_set_metadata.csv')
a = a.set_index('object_id')
b = b.set_index('object_id')

e = a.join(b)
# columns:
# columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
# object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

for passband in bands:
	hashes = defaultdict(list)
	times = {}

	for object_id, object_data in e.groupby(e.index.get_level_values(0)):
		mask = object_data['passband'] == passband
		flux = object_data['flux'][mask]
		flux_error = object_data['flux_err'][mask]

		time = object_data['mjd'][mask].values
		time_hash = (len(time), str(time[0]), str(time[-1]))
		if object_id == 615 or object_id == 1124:
			print(object_id, time_hash, passband)
		hashes[time_hash].append(object_id)
		times[time_hash] = time
	
	break
# for each object group:
# compute features
#for t in times.keys():



