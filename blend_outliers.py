from __future__ import print_function, division
"""
Combine classifier predictions with outlier prediction

Parameters:

PRIOR_STRENGTH		weight of prior (0: only take predictor, 1: only take prior)
PRIOR_FLATNESS		prior flatness (1: scale with training distribution, 0: flat over classes)
SPECZERR		fraction of wrong gal/exgal classification
EXPO			exponent on classifier probability
OUTLIER_CONF		strength of outlier votes (0: no outliers marked, 1: detected outliers are marked as probable as all non-outlier classes together, 3: outliers are marked as 3x as probable as all non-outlier classes together)
PRIOR_STRENGTH_OUTLIERS	additional outlier probability floor (0: take outlier detection directly)


"""

import sys, os
import numpy
import pandas

flatness = float(os.environ.get('PRIOR_FLATNESS', '1.0'))

def flatten(N, flatness):
	return numpy.where(N == 0, 0, N**flatness)
	if flatness >= 0:
		return N**(flatness + 1e-30)
	else:
		return numpy.where(N == 0, 0, N**flatness)

data_input = pandas.read_csv('training_set_metadata.csv')
classes = [6,15,16,42,52,53,62,64,65,67,88,90,92,95,99]
N = numpy.array([(data_input.target == cl).sum() for cl in classes], dtype='f')
N = flatten(N, flatness)
N[-1] = N[N>0].min()
Ntotal = N.sum()

print("priors:")
print("all:  ", ' '.join(['%.2f' % r for r in N / Ntotal]))
print()


print('loading prediction to correct, "%s" ...' % (sys.argv[1]))
df = pandas.read_csv(sys.argv[1])
w = float(os.environ.get('PRIOR_STRENGTH', '0.0'))
w_outliers = float(os.environ.get('PRIOR_STRENGTH_OUTLIERS', '0.0'))
w_wrongspecz = float(os.environ.get('SPECZERR', '0.01'))
expo = float(os.environ.get('EXPO', '1.0'))

outlier_voteweights = {
	'NORM-EllEnvelope':1, 
	'MM-EllEnvelope':1, 
	'NORM-IsolForest':10,
	'MM-IsolForest':10,
}
outlier_confidence = float(os.environ.get('OUTLIER_CONF', '1.0'))

outlier_votes = 0
nvotes_total = 0
for outliertechnique in 'NORM-EllEnvelope', 'NORM-IsolForest', 'MM-EllEnvelope', 'MM-IsolForest':
	for thresh in 0.001, 0.01, 0.04, 0.1, 0.4:
		print('loading outlier votes of %s[%s] ...' % (outliertechnique, thresh))
		is_outlier = -1 == numpy.loadtxt('test_set.csv.gz_novel_%s-%s.csv' % (outliertechnique, thresh))
		outlier_votes = outlier_votes + is_outlier * outlier_voteweights[outliertechnique]
		nvotes_total += outlier_voteweights[outliertechnique]

total_probs = 0

for col, Ni, Ngali, Nexgali in zip(df.columns[1:], N, Ngal, Nexgal):
	print('  adjusting column "%s" ...' % (col))
	if col == 'class_99':
		# novelty class: the other probabilities should sum to 1
		# here we add outlier vote as a factor relative to the other probabilities
		predictor = total_probs * outlier_votes * outlier_confidence / nvotes_total
		w_prior = w_outliers
		
		col_prob = predictor + w_prior
	else:
		w_prior = w
		predictor = df.loc[:,col]**expo
		
		prior_gal = Ngali * 1. / Ntotal_gal
		prior_exgal = Nexgali * 1. / Ntotal_exgal
		prior = Ni * 1. / Ntotal
		
		# blend in the prior
		prior = w_wrongspecz * numpy.where(~galmask, prior_gal, prior_exgal) + (1 - w_wrongspecz) * numpy.where(galmask, prior_gal, prior_exgal)
	
		col_prob = predictor * (1 - w_prior) + prior * w_prior
	
	total_probs = total_probs + col_prob
	df.loc[:,col] = col_prob

print('normalising columns ...')
for col in df.columns[1:]:
	df.loc[:,col] /= total_probs

print("writing data ...")
#data1.values[:,1:] = data1.values[:,1:]**2 * (1 - w) + (N * 1. / Ntotal).reshape((1,-1)) * w
df.to_csv(sys.argv[1] + '_blend_expo%s_noveltyconf%sprior%s_z%s_prior%sflat%s.csv.gz' % (expo, outlier_confidence, w_outliers, w_wrongspecz, w, flatness), 
	float_format='%.3e', index=False, header=True, compression='gzip', chunksize=100000)


