from __future__ import print_function, division
from collections import defaultdict
import os
import numpy
import pandas
from collections import defaultdict

#a = pandas.read_csv('training_set.csv')
b = pandas.read_csv('training_set_metadata.csv')
#a = a.set_index('object_id')
b = b.set_index('object_id')

targets  = numpy.unique(b.target.values)
nhave    = {t:(b.target == t).sum() for t in targets}
nhaveddf = {t:numpy.logical_and(b.ddf == 1, b.target == t).sum() for t in targets}
ntarget  = 400
uppertarget  = 1000
oversample_rates = {target:min(20, max(0, int((ntarget - nhave[target] + 0.2) / nhaveddf[target]))) for target in targets}
hypersample_rates = {target: max(0, (ntarget - nhave[target] - nhaveddf[target] * oversample_rates[target]) / nhave[target]) for target in targets}
undersample_rates = {target: uppertarget * 1. / nhave[target] for target in targets}


print("number of objects in each class:")
print(' '.join(['%4d' % nhave[target] for target in targets]))
print("number of DDF objects in each class:")
print(' '.join(['%4d' % nhaveddf[target] for target in targets]))
print("undersampling factor for each class:")
print(' '.join(['%.2f' % undersample_rates[target] for target in targets]))
print("oversampling factor for each class:")
print(' '.join(['%4d' % oversample_rates[target] for target in targets]))
print("hypersampling for each class:")
print(' '.join(['%.2f' % hypersample_rates[target] for target in targets]))

stream_meta = open('training_set_metadata.csv')
outstream_meta = open('resampled_training_set_metadata.csv', 'w')
outstream_meta.write(stream_meta.readline())

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
			hypersample = False
			if not is_ddf:
				hypersample = hypersample_rates[target]
				noversample = 0
			
			out_groups = [(last_object_id, input_lines)]
			j = 1
			
			for i in range(noversample):
				# select randomly 1-3 observations with a band of 3 nights, choose those nights.
				nobs = 2
				iobs_sel = numpy.random.randint(0, len(time_all), size=nobs)
				mask_sel = numpy.zeros(len(time_all), dtype=bool)
				for ti in time_all[iobs_sel]:
					mask_sel[numpy.logical_and(ti < time_all + 2.5, ti > time_all - 0.5)] = True
				if mask_sel.sum() > 4:
					iobs_sel = numpy.where(mask_sel)[0]
					out_groups.append((str(1000000000 * (j+1) + int(last_object_id)), [input_lines[i] for i in iobs_sel]))
					j += 1
			
			if hypersample > 0:
				last_ti = 0
				#for iobs_sel in range(len(time_all)):
				for iobs_sel in numpy.unique(numpy.random.randint(len(time_all), size=numpy.random.poisson(hypersample))):
					# remove a single night from a non-DDF observation
					ti = time_all[iobs_sel]
					if ti < last_ti + 0.5:
						continue
					ti = last_ti
					mask_sel = ~numpy.logical_and(ti < time_all + 0.5, ti > time_all - 0.5)
					if mask_sel.sum() > 4:
						iobs_sel = numpy.where(mask_sel)[0]
						out_groups.append((str(1000000000 * (j+1) + int(last_object_id)), [input_lines[i] for i in iobs_sel]))
						j += 1
			
			if numpy.random.uniform() > undersample_rates[target]:
				out_groups = []
			
			for prefix, lines in out_groups:
				for line in lines:
					outstream_data.write(prefix + line)
				if prefix == last_object_id:
					outstream_meta.write(prefix + current_meta_line)
				else:
					# set DDF to 0
					outstream_meta.write(prefix + current_meta_line.replace(',1,', ',0,'))
		
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


