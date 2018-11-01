from __future__ import print_function, division
import os, sys
import numpy
import pandas
import matplotlib.pyplot as plt

for filename in sys.argv[1:]:
	print('reading file "%s" ...' % filename)
	a = pandas.read_csv(filename, usecols=['object_id', 'mjd'])
	print('analysing file "%s" ...' % filename)
	label = filename.replace('.csv', '').replace('.gz', '')
	a = a.set_index('object_id')
	alldeltas = []
	allnobs = []

	for object_id, object_data in a.groupby(a.index.get_level_values(0)):
		time = object_data['mjd'].values
		deltat = time[1:] - time[:-1]
		nobs = len(time)
		mask = deltat > 0.5
		alldeltas += deltat[mask].tolist()
		allnobs.append(nobs)

	print('plotting ...')
	plt.figure('cadence')
	plt.hist(alldeltas, bins=1000, density=True, cumulative=True, histtype='step', label=label)
	plt.legend(loc='best')
	plt.ylabel('cumulative fraction')
	plt.xlabel('time [d]')
	plt.savefig('cadence.pdf')

	plt.figure('nobs')
	plt.hist(allnobs, bins=1000, density=True, cumulative=True, histtype='step', label=label)
	plt.legend(loc='best')
	plt.ylabel('cumulative fraction')
	plt.xlabel('number of observations')
	plt.savefig('cadence-nobs.pdf')


