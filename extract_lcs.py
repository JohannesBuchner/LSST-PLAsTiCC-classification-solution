from collections import defaultdict
import os
import numpy
import pandas
import matplotlib.pyplot as plt
#import FATS
#import tsfresh

## ugrizY EBV effect, assuming a flat spectrum
#extinction_factors = [0.0092394833, 0.0336530868, 0.0938432007, 0.1868623254, 0.2921024403, 0.4306847762]
bands = range(6)

a = pandas.read_csv('training_set.csv')
b = pandas.read_csv('training_set_metadata.csv')
a = a.set_index('object_id')
b = b.set_index('object_id')

plt.hist(b.target, bins=numpy.arange(101), histtype='step')
plt.hist(b.target[b.ddf == 1], bins=numpy.arange(101), histtype='step')
plt.yscale('log')
plt.savefig('target_hist.pdf', bbox_inches='tight')
plt.close()

# focus on one object for testing
#a = a[a.index == 615]
#b = b[b.index == 615]

e = a.join(b)

# columns:
# columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
# object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

targets = defaultdict(list)

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	if object_data['ddf'].values[0] == 0:
		continue
	target = object_data['target'].values[0]
	prefix = '%d' % target
	if len(targets[target]) > 10: 
		continue
	
	print(target, object_id)
	if not os.path.exists(prefix): os.mkdir(prefix)
	targets[target].append(object_id)
	
	for passband in bands:
		mask = numpy.logical_and(object_data['passband'] == passband, object_data['detected'] == 1)
		if mask.sum() < 2: continue
		flux = object_data['flux'][mask]
		flux_error = object_data['flux_err'][mask]
		time = object_data['mjd'][mask]
		## flux_obs = flux_intrinsic * extinction_factors * ebv
		#flux = flux_obs / (extinction_factors * ebv)
		
		# create features
		#print(time, flux, flux_error)
		numpy.savetxt('%s/lc%d-%d.txt' % (prefix, object_id, passband), 
			numpy.transpose([time.values, flux.values, flux_error.values]))
		
	# split already into subclasses:
	#galactic = numpy.isnan(object_data['distmod'].values[0])

# for each object:
# get ebv
# correct the flux
# compute features




