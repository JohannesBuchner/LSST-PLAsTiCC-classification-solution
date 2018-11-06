from __future__ import print_function, division
from collections import defaultdict
import os, sys
import numpy
import scipy.stats
import pandas
from numpy import pi, exp, log10

transform_SED_info = os.environ.get('SEDTRANSFORMER', '0') == '1'

bands = range(6)
# very roughly read off the LSST passband plot
wavelengths = [357, 475, 621, 755, 879, 1000]
passband_efficiencies = [65, 140, 131, 118, 98, 151]

wavelengths = numpy.array(wavelengths)
passband_efficiencies = numpy.array(passband_efficiencies)

prefix = sys.argv[1]
a = pandas.read_csv(prefix + '.csv')
b = pandas.read_csv(prefix + '_metadata.csv')
a = a.set_index('object_id')
b = b.set_index('object_id')

# slim down table
for col in 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'distmod', 'mwebv':
	try:
		b.pop(col)
	except KeyError:
		pass

e = a.join(b)

# columns:
flux_columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

del a, b

if not transform_SED_info:
	fout = open(prefix + '_SED_features.txt', 'w')
	header = "uflux,gflux,rflux,iflux,zflux,Yflux,photoz,photozerr,dtpeak,weight,target"
	fout.write(header + "\n")
	linefmt = ('%f,' * (header.count(','))).rstrip(',') + ",%d\n"
else:
	import train_SED
	clf = train_SED.get_SED_transformer()
	fout = open(prefix + '_SEDprob_features.txt', 'w')
	header = ','.join("SEDprob%d" % i for i in train_SED.labels)
	linefmt = ','.join("%.2f" for i in train_SED.labels) + "\n"
	Nprobfeatures = len(train_SED.labels)
	fout.write(header + "\n")
	prior_proba = numpy.ones(Nprobfeatures) / Nprobfeatures
	
	def compute_probability(samples):
		if len(samples) == 0:
			return prior_proba
		# drop target and weight
		X = numpy.array(samples)[:,:-2]
		#print("compute_probability for:", X.shape)
		X[X<-4] = -4
		X = train_SED.mytransformer.transform(X)
		proba = clf.predict_proba(X)
		# here we do a maximum over the probabilities
		# this should make the most confident classification dominate,
		# but if there is diversity, it will also be reflected
		proba = numpy.max(proba, axis=0)
		return proba
	
	

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	#specz = object_data['hostgal_specz'].values[0]
	photoz = object_data['hostgal_photoz'].values[0]
	photoz_error = object_data['hostgal_photoz_err'].values[0]
	if transform_SED_info:
		target = 0
	else:
		target = object_data['target'].values[0]
	
	all_time = object_data['mjd'].values
	all_flux = object_data['flux'].values
	#all_flux_error = object_data['flux_err'].values
	all_passband = object_data['passband'].values
	is_detected = object_data['detected'] == 1
	
	tpeak = all_time[numpy.argmax(all_flux)]
	lastt = -100
	deltat = 0.1
	samples = []
	for t in numpy.unique(all_time[numpy.logical_and(all_flux > 0, is_detected)]):
		if t < lastt + deltat:
			continue
		lastt = t
		mask = numpy.logical_and(numpy.logical_and(all_time >= t, all_time < t + deltat), 
			numpy.logical_and(all_flux > 0, is_detected))
		passband = all_passband[mask]
		flux = all_flux[mask] / all_flux[mask].max()
		
		args = []
		for band in bands:
			mask_band = passband == band
			if mask_band.any():
				args.append(log10(flux[numpy.where(mask_band)[0][0]]))
			else:
				args.append(-99)
		samples.append(args + [photoz, photoz_error, t - tpeak, 0, target])
	
	if not transform_SED_info:
		for features in samples:
			features[-2] = 1. / len(samples)
			fout.write(linefmt % tuple(features))
	else:
		results = compute_probability(samples)
		#print(results.shape, linefmt)
		fout.write(linefmt % tuple(results))
		
	


