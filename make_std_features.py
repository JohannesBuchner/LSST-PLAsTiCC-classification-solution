from __future__ import print_function, division
import os
import numpy
import scipy.stats
import pandas
import sys
from astropy.stats import LombScargle

def LC_features(time, flux, flux_error):
	if len(time) < 3:
		return [-99] * 10
	mag = numpy.log10(flux) * 0.4
	#lc = numpy.array([mag, time])
	variance = mag.std()
	skew = scipy.stats.skew(mag)
	kurtosis = scipy.stats.kurtosis(mag)
	iqr = scipy.stats.iqr(mag)
	shapiro_wilk, _p_value = scipy.stats.shapiro(mag)
	fracabove = (mag > mag.mean()).mean()

	frequency, power = LombScargle(time, flux, flux_error).autopower()
	best_frequency = frequency[numpy.argmax(power)]
	LS_period = 1.0 / best_frequency
	power0 = power[0]
	a, b, c = numpy.argsort(power)[::-1][:3]
	power1 = power[a]
	power2 = power[b]
	power3 = power[c]
	R21 = power2 / power1
	R31 = power3 / power1
	R01 = power.sum() / power1

	features = [
		variance, skew,
		kurtosis, iqr, shapiro_wilk, fracabove, 
		LS_period, R21, R31, R01
	]
	return features




## ugrizY EBV effect, assuming a flat spectrum
#extinction_factors = [0.0092394833, 0.0336530868, 0.0938432007, 0.1868623254, 0.2921024403, 0.4306847762]
bands = range(6)

prefix = sys.argv[1]
a = pandas.read_csv(prefix + '.csv')
b = pandas.read_csv(prefix + '_metadata.csv')
a = a.set_index('object_id')
b = b.set_index('object_id')

e = a.join(b)

# columns:
flux_columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

fout = open(prefix + '_std_features.txt', 'w')
fout.write("#")
for color in 'ugrizY':
	fout.write("%s_nmeasurements,%s_ngoodmeasurements,%s_goodtimerange," % (color, color, color))
	for c in "variance, skew, kurtosis, iqr, shapiro_wilk, fracabove, LS_period, R21, R31, R01".split(', '):
		fout.write("%s_%s," % (color, c))
fout.write("\n")

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	print(object_id)
	
	lc_features_all = []
	for passband in bands:
		nmeasurements = (object_data['passband'] == passband).sum()
		mask = numpy.logical_and(object_data['passband'] == passband, 
			numpy.logical_and(object_data['flux'] > 0, object_data['detected'] == 1))
		
		flux = object_data['flux'][mask].values
		flux_error = object_data['flux_err'][mask].values
		time = object_data['mjd'][mask].values
		
		# create features
		lc_features = LC_features(time, flux, flux_error)
		lc_features_all += [nmeasurements, len(time), time[-1] - time[0]]
		lc_features_all += lc_features
		
		#print(lc_slope_features)
		#print()
	
	#break
	fout.write(('%f,' * len(lc_features_all)) % tuple(lc_features_all))
	fout.write('\n')



