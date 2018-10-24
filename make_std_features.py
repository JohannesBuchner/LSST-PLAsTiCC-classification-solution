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

def get_standard_stats(mag):
	if len(mag) == 0:
		return [numpy.nan]*5
	elif len(mag) == 1:
		medianmag = numpy.median(mag)
		return [medianmag, 0.0, 0.0, -3.0, 0.0]
	else:
		medianmag = numpy.median(mag)
		variance = mag.std()
		skew = scipy.stats.skew(mag) if len(mag) > 2 else 0.0
		kurtosis = scipy.stats.kurtosis(mag)
		iqr = scipy.stats.iqr(mag)
		return [medianmag, variance, skew, kurtosis, iqr]

def LC_features(time, flux, flux_error):
	nmeasurements = len(time)
	totaltimedetected = time[-1] - time[0] if len(time) > 1 else 0
	features = [nmeasurements, totaltimedetected]
	
	mag = numpy.log10(flux) * 0.4
	#lc = numpy.array([mag, time])
	maxmag = numpy.max(mag) if len(mag) > 0 else numpy.nan
	
	medianmag, variance, skew, kurtosis, iqr = get_standard_stats(mag)
	
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
	
	features += get_standard_stats(time)[1:] # skipping median time
	
	deltat = time[1:] - time[:-1]
	features += get_standard_stats(deltat)
	
	slopes = (mag[1:] - mag[:-1]) / (time[1:] - time[:-1])
	features += get_standard_stats(slopes)
	
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
	features += get_standard_stats(slopes[:i+1])
	features += get_standard_stats(slopes[i:])
	
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

# slim down table
for col in 'ra', 'decl', 'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz', 'hostgal_photoz_err', 'distmod', 'mwebv', 'target':
	try:
		b.pop(col)
	except KeyError:
		pass

e = a.join(b)
del a, b

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
	all_detected = object_data['detected'].values == 1
	all_time = object_data['mjd'].values
	all_fluxes = object_data['flux'].values
	all_flux_errors = object_data['flux_err'].values
	all_passbands = object_data['passband'].values
	
	allnmeasurements = len(all_fluxes)
	was_detected = numpy.logical_and(all_fluxes > 0, all_detected)
	allgoodmeasurements = was_detected.sum()
	if allgoodmeasurements > 0:
		alltimerange = all_time[was_detected].max() - all_time[was_detected].min()
	else:
		alltimerange = 0
	
	lc_features_all = []
	for passband in bands:
		mask = all_passbands == passband
		
		nmeasurements = mask.sum()
		was_detected_run = was_detected[mask]
		# remove non-detections in beginning and end
		was_detected_run = numpy.trim_zeros(was_detected_run)
		avgdetrun, stddetrun = make_runstats(was_detected_run)
		allavgdetrun.append(avgdetrun)

		mask = numpy.logical_and(mask, was_detected)
		time = all_time[mask]
		flux = all_fluxes[mask]
		flux_error = all_flux_errors[mask]
		
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



