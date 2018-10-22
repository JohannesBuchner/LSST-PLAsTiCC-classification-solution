from __future__ import print_function, division
import os
import numpy
import scipy.stats
import pandas
import sys
from astropy.stats import LombScargle
import itertools

def runs_of_ones(bits):
	return [sum(group) for bit, group in itertools.groupby(bits) if bit]

def make_runstats(mask):
	runlengths = runs_of_ones(mask)
	return numpy.mean(runlengths), numpy.std(runlengths)

def LC_features(time, flux, flux_error):
	nmeasurements = len(time)
	totaltimedetected = time[-1] - time[0] if len(time) > 1 else 0
	features = [nmeasurements, totaltimedetected]
	
	mag = numpy.log10(flux) * 0.4
	#lc = numpy.array([mag, time])
	maxmag = numpy.max(mag) if len(mag) > 0 else numpy.nan
	medianmag = numpy.median(mag)
	variance = mag.std()
	skew = scipy.stats.skew(mag)
	kurtosis = scipy.stats.kurtosis(mag)
	iqr = scipy.stats.iqr(mag)
	fracabove = (mag > mag.mean()).mean()
	if len(mag) < 3:
		shapiro_wilk, _p_value = numpy.nan, numpy.nan

		LS_period = numpy.nan
		R21 = numpy.nan
		R31 = numpy.nan
		R01 = numpy.nan
	else:
		shapiro_wilk, _p_value = scipy.stats.shapiro(mag)

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

	features += [
		medianmag, maxmag,
		variance, skew,
		kurtosis, iqr, shapiro_wilk, fracabove, 
		LS_period, R21, R31, R01
	]
	
	tvariance = time.std()
	tskew = scipy.stats.skew(time)
	tkurtosis = scipy.stats.kurtosis(time)
	tiqr = scipy.stats.iqr(time)
	
	deltat = time[1:] - time[:-1]
	dtmedian = numpy.median(deltat)
	dtvariance = deltat.std()
	dtskew = scipy.stats.skew(deltat)
	dtkurtosis = scipy.stats.kurtosis(deltat)
	dtiqr = scipy.stats.iqr(deltat)
	
	slopes = (mag[1:] - mag[:-1]) / (time[1:] - time[:-1])
	dsmedian = numpy.median(slopes)
	dsvariance = slopes.std()
	dsskew = scipy.stats.skew(slopes)
	dskurtosis = scipy.stats.kurtosis(slopes)
	dsiqr = scipy.stats.iqr(slopes)
	
	features += [
		tvariance, tskew, tkurtosis, tiqr,
		dtmedian, dtvariance, dtskew, dtkurtosis, dtiqr,
		dsmedian, dsvariance, dsskew, dskurtosis, dsiqr,
	]

	# count dips: where the flux is lower than earlier and later
	mask_left_down  = flux[1:-1] + flux_error[1:-1] < (flux[0:-2] - flux_error[0:-2]) * 0.8
	mask_right_up   = flux[1:-1] + flux_error[1:-1] < (flux[2:]   - flux_error[2:]  ) * 0.8
	ndips = numpy.logical_and(mask_left_down, mask_right_up).sum()
	mask_left_up    = flux[1:-1] - flux_error[1:-1] > (flux[0:-2] + flux_error[0:-2]) / 0.8
	mask_right_down = flux[1:-1] - flux_error[1:-1] > (flux[2:]   + flux_error[2:]  ) / 0.8
	npeaks = numpy.logical_and(mask_left_up, mask_right_down).sum()
	mask_down = flux[1:] - flux_error[1:] < flux[:-1] + flux_error[:-1]
	mask_up   = flux[1:] + flux_error[1:] > flux[:-1] - flux_error[:-1]
	
	nmeasurements = len(flux)
	avguprun, stduprun = make_runstats(mask_up)
	avgdownrun, stddownrun = make_runstats(mask_down)
	
	features += [
		ndips, npeaks, 
		avguprun, stduprun,
		avgdownrun, stddownrun,
	]
	
	# slopes before and after peak
	i = numpy.argmax(flux) if len(flux) > 0 else 0
	dsmedian = numpy.median(slopes[:i+1])
	dsvariance = slopes[:i+1].std()
	dsskew = scipy.stats.skew(slopes[:i+1])
	dskurtosis = scipy.stats.kurtosis(slopes[:i+1])
	dsiqr = scipy.stats.iqr(slopes[:i+1])
	features += [dsmedian, dsvariance, dsskew, dskurtosis, dsiqr]
	dsmedian = numpy.median(slopes[i:])
	dsvariance = slopes[i:].std()
	dsskew = scipy.stats.skew(slopes[i:])
	dskurtosis = scipy.stats.kurtosis(slopes[i:])
	dsiqr = scipy.stats.iqr(slopes[i:])
	features += [dsmedian, dsvariance, dsskew, dskurtosis, dsiqr]

	return features


## ugrizY EBV effect, assuming a flat spectrum
#extinction_factors = [0.0092394833, 0.0336530868, 0.0938432007, 0.1868623254, 0.2921024403, 0.4306847762]
bands = range(6)

prefix = sys.argv[1]
print("reading data...")
a = pandas.read_csv(prefix + '.csv')
print("reading metadata...")
b = pandas.read_csv(prefix + '_metadata.csv')
print("processing...")
a = a.set_index('object_id')
b = b.set_index('object_id')

e = a.join(b)

# columns:
flux_columns = ['mjd', 'passband', 'flux', 'flux_err', 'detected']
object_columns = ['ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target']

fout = open(prefix + '_std_features.txt', 'w')
header = ''
bandcolumns = """
medianmag, maxmag,
variance, skew,
kurtosis, iqr, shapiro_wilk, fracabove, 
LS_period, R21, R31, R01,
tvariance, tskew, tkurtosis, tiqr,
dtmedian, dtvariance, dtskew, dtkurtosis, dtiqr,
dsmedian, dsvariance, dsskew, dskurtosis, dsiqr,
ndips, npeaks, avguprun, stduprun, avgdownrun, stddownrun,
dsupmedian, dsupvariance, dsupskew, dsupkurtosis, dsupiqr,
dsdownmedian, dsdownvariance, dsdownskew, dsdownkurtosis, dsdowniqr
"""

for color in 'ugrizY':
	header += "%s_nmeasurements,%s_ngoodmeasurements,%s_goodtimerange," % (color, color, color)
	for c in bandcolumns.replace('\n','').replace(' ', '').split(','):
		header += "%s_%s," % (color, c)
	header += "%s_avgdetrun,%s_stddetrun," % (color, color)
header += "all_nmeasurements,all_ngoodmeasurements,all_timerange,all_avgdetrun"
#print("header columns: %d" % header.count(','))
#print(header)
fout.write(header + '\n')
linefmt = ('%f,' * (header.count(',')+1)).rstrip(',') + "\n"

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	print(object_id)
	
	allavgdetrun = []
	allnmeasurements = len(object_data)
	was_detected = numpy.logical_and(object_data['flux'] > 0, object_data['detected'] == 1)
	allgoodmeasurements = was_detected.sum()
	all_time = object_data['mjd'].values
	if was_detected:
		alltimerange = all_time[was_detected].max() - all_time[was_detected].min()
	else:
		alltimerange = 0
	lc_features_all = []
	for passband in bands:
		nmeasurements = (object_data['passband'] == passband).sum()
		mask = object_data['passband'] == passband
		time = object_data['mjd'][mask].values
		flux = object_data['flux'][mask].values
		flux_error = object_data['flux_err'][mask].values
		was_detected = numpy.logical_and(object_data['flux'][mask] > 0, object_data['detected'][mask] == 1)
		# remove non-detections in beginning and end
		was_detected = numpy.trim_zeros(was_detected)
		avgdetrun, stddetrun = make_runstats(was_detected)
		allavgdetrun.append(avgdetrun)

		mask = numpy.logical_and(object_data['passband'] == passband, 
			numpy.logical_and(object_data['flux'] > 0, object_data['detected'] == 1))
		
		flux = object_data['flux'][mask].values
		flux_error = object_data['flux_err'][mask].values
		time = object_data['mjd'][mask].values
		
		# create features
		lc_features = LC_features(time, flux, flux_error)
		lc_features_all += [nmeasurements]
		lc_features_all += lc_features
		lc_features_all += [avgdetrun, stddetrun]
		
		#print(lc_slope_features)
		#print()
	
	allavgdetrun = numpy.mean(allavgdetrun)
	lc_features_all += [allnmeasurements, allgoodmeasurements, alltimerange, allavgdetrun]
	
	#print("   ",len(lc_features_all))
	#break
	fout.write(linefmt % tuple(lc_features_all))



