from __future__ import print_function, division
from collections import defaultdict
import os
import numpy
import scipy.stats
import pandas
import sys

c2 = numpy.polynomial.chebyshev.Chebyshev((0,0,1))
def compute_concavity(x, y, yerr, ymodel):
	resid = y - ymodel
	rel_resid = (y - ymodel) / ymodel
	# normalize to -1 to +1 interval
	xnorm = (x - x[0]) * 2 / (x[-1] - x[0]) - 1
	# variance beyond noise
	excess_variance = (resid**2 - yerr**2).sum() / (ymodel**2).sum()
	excess_variance = ((resid/yerr)**2).mean()
	
	# concavity: weighting to edges
	c2_values = (c2(xnorm) + 1) / 2
	concavity = (c2_values * rel_resid).sum() / c2_values.sum()
	
	return excess_variance, concavity

def fit_logline(time, flux, flux_error):
	x, y, y_err = time, numpy.log10(flux), numpy.log10(flux_error)
	slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
	excess_variance, concavity = compute_concavity(time, flux, flux_error, ymodel=10**(slope * x + intercept))

	t = numpy.linspace(x[0], x[-1], 400)
	return [slope, intercept, excess_variance, concavity]

def get_indices(time):
	# bootstrap
	return numpy.unique(numpy.random.randint(0, len(time), size=len(time)))

def LC_slopes(time, flux, flux_error, specz, photoz, photoz_error):
	breaktimes = []
	slopes = []
	# cross-validate or bootstrap
	for bs in range(20):
		if specz == 0:
			z = 0
		elif numpy.isfinite(specz):
			z = specz
		else:
			z = numpy.random.normal(photoz, photoz_error)
		
		if len(time) > 0:
		
			i = get_indices(time)
			# get subsampled restframe time
			# at rest frame, more time passed
			time_bs = time[i] / (1 + z)
			flux_bs_resampled = numpy.random.normal(flux[i], flux_error[i])
			flux_bs, flux_error_bs = flux[i], flux_error[i]
		
			# find maximum
			peak = numpy.argmax(flux_bs)
			peakflux = flux_bs[peak]
		else:
			time_bs = [0]
		
			# find maximum
			peak = 0
			peakflux = 1e10
		
		# fit left part
		lresult = [numpy.random.normal(0, 5), time_bs[peak], 10, 10]
		if len(time_bs[0:peak+1]) > 2:
			lresult = fit_logline(time_bs[0:peak+1], flux_bs[0:peak+1], flux_error_bs[0:peak+1])
		
		# fit right part
		rresult = [numpy.random.normal(0, 5), time_bs[peak], 10, 10]
		if len(time_bs[peak:]) > 2:
			rresult = fit_logline(time_bs[peak:], flux_bs[peak:], flux_error_bs[peak:])
		
		# find intercept
		# lresult[0] * t + lresult[1] == rresult[0] * t + rresult[1]
		# (lresult[0] - rresult[0]) * t == rresult[1] - lresult[1]
		# t = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
		tintercept = (rresult[1] - lresult[1]) / (lresult[0] - rresult[0])
		peak_intercept = lresult[0] * tintercept + lresult[1]
		
		slopes.append(lresult + rresult + [numpy.log10(peakflux), numpy.log10(peak_intercept)])
		breaktimes.append(tintercept)

	#for entry in slopes:
	#	for v in entry:
	#		print('%.3f' % v, end='\t')
	#	print()
	
	slopes = numpy.asarray(slopes)
	goodl = slopes[:,2] > 2
	goodr = slopes[:,2+4] > 2
	
	# features:
	# fraction of left good fits
	# fraction of right good fits
	# (for each of the following: 25%, 50%, 75% quartiles)
	# left poor fits slope 
	# left good fits slope
	# left poor fits concavity 
	# left good fits concavity
	# right poor fits slope 
	# right good fits slope
	# right poor fits concavity 
	# right good fits concavity
	# left fits variance
	# right fits variance
	# peak flux
	# peak flux loglinear interpolated
	results = []
	results += [goodl.mean(), goodr.mean()]
	for v in slopes[goodl,0], slopes[~goodl,0], slopes[goodl,3], slopes[~goodl,3], slopes[goodr,0+4], slopes[~goodr,0+4], slopes[goodr,3+4], slopes[~goodr,3+4], slopes[:,2], slopes[:,4], slopes[:,8], slopes[:,9]:
		vmask = numpy.isfinite(v)
		if vmask.any():
			results += scipy.stats.mstats.mquantiles(v[vmask], [0.25, 0.5, 0.75]).tolist()
		else:
			results += [-99,-99,-99]
	return results

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

fout = open(prefix + '_slope_features.txt', 'w')
# left poor fits slope 
# left good fits slope
# left poor fits concavity 
# left good fits concavity
# right poor fits slope 
# right good fits slope
# right poor fits concavity 
# right good fits concavity
# left fits variance
# right fits variance
# peak flux
# peak flux loglinear interpolated

fout.write("#")
for color in 'ugrizY':
	fout.write("%s_fraclgf, %s_fracrgf, " % (color, color))
	for c in "lpfslope, lgfslope, lpfconc, lgfconc, rpfslope, rgfslope, rpfcconc, rgfconc, lfvar, rfvar, peakflux, peakfluxi".split(', '):
		fout.write("%s_%s_25, %s_%s_50, %s_%s_75, " % (color, c, color, c, color, c))
	fout.write("%s_peakcolor, %s_peakcolori, " % (color, color))
fout.write("\n")

for object_id, object_data in e.groupby(e.index.get_level_values(0)):
	print(object_id)
	specz = object_data['hostgal_specz'].values[0]
	photoz = object_data['hostgal_photoz'].values[0]
	photoz_error = object_data['hostgal_photoz_err'].values[0]
	all_time = object_data['mjd']
	all_flux = object_data['flux']
	all_flux_error = object_data['flux_err']
	all_passband = object_data['passband']
	is_detected = object_data['detected'] == 1
	lc_slope_features_all = []
	for passband in bands:
		mask = numpy.logical_and(all_passband == passband, 
			numpy.logical_and(all_flux > 0, is_detected))
		
		flux = all_flux[mask].values
		flux_error = all_flux_error[mask].values
		time = all_time[mask].values
		
		# create features
		lc_slope_features = LC_slopes(time, flux, flux_error, specz, photoz, photoz_error)
		lc_slope_features_all.append(lc_slope_features)
		
		#print(lc_slope_features)
		#print()
	
	lc_slope_features_all = numpy.asarray(lc_slope_features_all)
	peak_colors = lc_slope_features_all[:,-2] - lc_slope_features_all[:,-2].max()
	peak_colors_intercept = lc_slope_features_all[:,-1] - lc_slope_features_all[:,-1].max()
	lc_slope_features_all = numpy.hstack((lc_slope_features_all.flatten(), peak_colors, peak_colors_intercept))
	
	fout.write(('%f,' * len(lc_slope_features_all)) % tuple(lc_slope_features_all))
	fout.write('\n')



