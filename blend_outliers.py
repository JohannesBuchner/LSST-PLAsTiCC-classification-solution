from __future__ import print_function, division
"""
Combine classifier predictions with outlier prediction

Parameters:

PRIOR_STRENGTH		weight of prior (0: only take predictor, 1: only take prior)
PRIOR_FLATNESS		prior flatness (1: scale with training distribution, 0: flat over classes)
EXPO			exponent on classifier probability
OUTLIER_CONF		strength of outlier votes (0: no outliers marked, 1: detected outliers are marked as probable as all non-outlier classes together, 3: outliers are marked as 3x as probable as all non-outlier classes together)
PRIOR_STRENGTH_OUTLIERS	additional outlier probability floor (0: take outlier detection directly)


"""

import sys, os
import numpy
import pandas

print('loading prediction to correct, "%s" ...' % (sys.argv[1]))
df = pandas.read_csv(sys.argv[1])
w = float(os.environ.get('PRIOR_STRENGTH', '0.0'))
w_outliers = float(os.environ.get('PRIOR_STRENGTH_OUTLIERS', '0.0'))
expo = float(os.environ.get('EXPO', '1.0'))

outliertechnique = os.environ.get('OUTLIER_METHOD') # one of NORM-EllEnvelope NORM-IsolForest MM-EllEnvelope MM-IsolForest
outlier_confidence = float(os.environ.get('OUTLIER_CONF', '1.0'))
classify_outliers = os.environ.get('OUTLIER_CLASSIFY', '1') == '1'

if outliertechnique is None:
	# no additional outliers
	outlier_votes = 0
	outlier_confidence = 0
	classify_outliers = True
else:
	print('loading outlier votes of %s ...' % (outliertechnique))
	outlier_ids = numpy.loadtxt('test_set.csv.gz_novel_%s.csv' % (outliertechnique), delimiter=',', dtype=int)[:,0]
	outlier_votes = numpy.zeros(len(df), dtype=bool)
	outlier_votes[outlier_ids] = True
	print('%.2f%% outliers' % (outlier_votes.mean()*100))

total_probs = 0

for col in df.columns[1:]:
	print('  adjusting column "%s" ...' % (col))
	if col == 'class_99':
		# novelty class: the other probabilities should sum to 1
		# here we add outlier vote as a factor relative to the other probabilities
		predictor = total_probs * outlier_votes * outlier_confidence
		w_prior = w_outliers
		
		col_prob = predictor + w_prior
	else:
		w_prior = w
		predictor = df.loc[:,col]**expo
		
		prior = 1.0
		
		col_prob = predictor * (1 - w_prior) + prior * w_prior
		
		if not classify_outliers:
			# use flat probabilities for outliers, because we do not know what they are
			col_prob = numpy.where(outlier_votes, 0.4, col_prob)
	
	total_probs = total_probs + col_prob
	df.loc[:,col] = col_prob

print('normalising columns ...')
for col in df.columns[1:]:
	df.loc[:,col] /= total_probs

if outliertechnique is None:
	filename = sys.argv[1] + '_blend_expo%sprior%s_outlierprior%s.csv.gz' % (expo, w, w_outliers)
else:
	filename = sys.argv[1] + '_blend_expo%sprior%s_%soutlierconf%sprior%s%s.csv.gz' % (expo, w, outliertechnique, outlier_confidence, w_outliers, '' if classify_outliers else 'noclass')
print('writing data to "%s"...' % filename)
#data1.values[:,1:] = data1.values[:,1:]**2 * (1 - w) + (N * 1. / Ntotal).reshape((1,-1)) * w
df.to_csv(filename, 
	float_format='%.3e', index=False, header=True, compression='gzip')


