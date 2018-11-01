from __future__ import print_function, division
import os
import numpy
import pandas

#a = pandas.read_csv('training_set.csv')
b = pandas.read_csv('training_set_metadata.csv')
#a = a.set_index('object_id')
b = b.set_index('object_id')
numpy.random.seed(1)

targets  = numpy.unique(b.target.values)
nhave    = {t:(b.target == t).sum() for t in targets}
nhaveddf = {t:numpy.logical_and(b.ddf == 1, b.target == t).sum() for t in targets}
ntarget  = 400
uppertarget  = 1000
oversample_rates = {target:min(20, max(3, int((ntarget - nhave[target] + 0.2) / nhaveddf[target]))) for target in targets}
#hypersample_rates = {target: max(0, (ntarget - nhave[target] - nhaveddf[target] * oversample_rates[target]) / nhave[target]) for target in targets}
#undersample_rates = {target: uppertarget * 1. / nhave[target] for target in targets}

# for resampling time cadence
a = pandas.read_csv('training_set.csv', usecols=['object_id', 'mjd'])
a = a.set_index('object_id')
alldeltas = []
for object_id, object_data in a.groupby(a.index.get_level_values(0)):
	time = object_data['mjd'].values
	deltat = time[1:] - time[:-1]
	alldeltas += deltat[deltat > 0.5].tolist()
del a
alldeltas = numpy.sort(alldeltas)



# for resampling redshift
c = pandas.read_csv('test_set_metadata.csv')
mask = c.hostgal_specz > 0
specz = c.hostgal_specz[mask].values
photoz = c.hostgal_photoz[mask].values
del c

Nz = 15
zindex = (numpy.log10(photoz+1) * Nz / 0.5).astype(int)
zindex[zindex <  0] = 0
zindex[zindex > Nz-1] = Nz-1
zdists = [[] for i in range(Nz)]
for zi, speczi in zip(zindex, specz):
	zdists[zi].append(speczi)
zdists = [numpy.sort(zdist) for zdist in zdists]

def sample_photoz(photozi, photozi_err, Nsamples=None):
	# draw from empirical distribution
	zindexi = int(numpy.log10(photozi+1) * Nz / 0.5)
	zindexi = max(0, min(Nz-1, zindexi))
	zdist = zdists[zindexi]
	znew = numpy.random.choice(zdist, replace=True, size=Nsamples)
	#print(photozi, '->', znew)
	return znew
	# original:
	# return numpy.random.normal(photozi, photozi_err, size=Nsamples)

print("number of objects in each class:")
print(' '.join(['%4d' % nhave[target] for target in targets]))
print("number of DDF objects in each class:")
print(' '.join(['%4d' % nhaveddf[target] for target in targets]))
#print("undersampling factor for each class:")
#print(' '.join(['%.2f' % undersample_rates[target] for target in targets]))
print("oversampling factor for each class:")
print(' '.join(['%4d' % oversample_rates[target] for target in targets]))
#print("hypersampling for each class:")
#print(' '.join(['%.2f' % hypersample_rates[target] for target in targets]))

stream_meta = open('training_set_metadata.csv')
outstream_meta = open('resampled_training_set_metadata.csv', 'w')
outstream_meta.write(stream_meta.readline())
def add_noise(metaline):
	_,ra,decl,gal_l,gal_b,ddf,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target = metaline.split(',')
	ra    = '%f' % max(0, min(360, numpy.random.normal(float(ra), 0.1)))
	decl  = '%f' % max(-180, min(180, numpy.random.normal(float(decl), 0.1)))
	gal_l = '%f' % max(0, min(360, numpy.random.normal(float(gal_l), 0.1)))
	gal_b = '%f' % max(-180, min(180, numpy.random.normal(float(gal_b), 0.1)))
	if float(hostgal_photoz) > 0:
		hostgal_photoz = '%f' % sample_photoz(float(hostgal_photoz), float(hostgal_photoz_err))
		hostgal_photoz_err = '%f' % max(0, numpy.random.normal(float(hostgal_photoz_err), 0.01))
	hostgal_specz = '0'
	distmod = '%f' % max(0, numpy.random.normal(float(distmod), 0.001))
	mwebv = '%f' % max(0, numpy.random.normal(float(mwebv), 0.01))
	
	return ',' + ','.join([ra,decl,gal_l,gal_b,ddf,hostgal_specz,hostgal_photoz,hostgal_photoz_err,distmod,mwebv,target])

def resample_data(input_lines):
	output_lines = []
	for input_line in input_lines:
		mjd, passband, flux, flux_err, detected = input_line[1:].split(',')
		flux = '%f' % (numpy.random.normal(float(flux), float(flux_err)))
		output_line = ',' + ','.join([mjd, passband, flux, flux_err, detected])
		output_lines.append(output_line)
	return output_lines

stream_data = open('training_set.csv')
outstream_data = open('resampled_training_set.csv', 'w')
outstream_data.write(stream_data.readline())

last_object_id = ''
current_meta_line = ''

input_times = []
input_lines = []

for line in stream_data:
	parts = line.split(',')
	object_id, mjdstr = parts[:2]
	if object_id != last_object_id:
		# new object encountered
		if last_object_id != "":
			# finalise previous
			time_all = numpy.array(input_times)

			noversample = oversample_rates[target]
			out_groups = [(last_object_id, input_lines)]
			j = 1
			
			last_ti = 0
			for i in range(noversample):
				# skip beginning or end
				t0 = time_all[0] + numpy.random.choice(alldeltas) - 0.1
				t1 = time_all[-1] - numpy.random.choice(alldeltas) - 0.1
				mask_sel = numpy.logical_and(time_all >= t0, time_all <= t1)
				if mask_sel.sum() > 4:
					iobs_sel = numpy.where(mask_sel)[0]
					out_groups.append((str(1000000000 * (j+1) + int(last_object_id)), resample_data([input_lines[i] for i in iobs_sel])))
					j += 1
			
			#if numpy.random.uniform() > undersample_rates[target]:
			#	out_groups = []
			
			for prefix, lines in out_groups:
				for line in lines:
					outstream_data.write(prefix + line)
				if prefix == last_object_id:
					outstream_meta.write(prefix + current_meta_line)
				else:
					# set DDF to 0
					outstream_meta.write(prefix + add_noise(current_meta_line.replace(',1,', ',0,')))
		
		# set to new object
		input_times = []
		input_lines = []
		last_object_id = object_id
		current_meta_line = stream_meta.readline()
		parts = current_meta_line.split(',')
		assert parts[0] == last_object_id, (parts[0], last_object_id)
		is_ddf = parts[5] == '1'
		target = int(parts[-1])
		current_meta_line = current_meta_line[current_meta_line.index(','):]
	
	mjd = float(mjdstr)
	input_lines.append(line[line.index(','):])
	input_times.append(mjd)

import matplotlib.pyplot as plt
c = pandas.read_csv('resampled_training_set_metadata.csv')
c = c.set_index('object_id')

plt.hist(b.target, bins=numpy.arange(101), histtype='step', linestyle='--', linewidth=0.2, color='k')
#plt.hist(b.target[b.ddf == 1], bins=numpy.arange(101), histtype='step', linewidth=0.1)
plt.hist(c.target, bins=numpy.arange(101), histtype='step', color='r')
#plt.hist(c.target[c.ddf == 1], bins=numpy.arange(101), histtype='step')
#plt.yscale('log')
plt.savefig('resampled_target_hist.pdf', bbox_inches='tight')
plt.close()


