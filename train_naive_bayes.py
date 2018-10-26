from __future__ import print_function, division
from alltrain import *
from numpy import exp, pi, sqrt, log10, log
import matplotlib.pyplot as plt

columns = train.columns[valid_column_mask]
del train

#X = mytransformer.fit_transform(X)

execute = unknown_data_file is not None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')
	unknown = unknown.values
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)
	#unknown = mytransformer.transform(unknown)
else:
	unknown = None


def plothist(x, label):
	x = x[numpy.isfinite(x)]
	if x.max() > x.min():
		plt.hist(x, histtype='step', normed=True, cumulative=True, label=label)
	else:
		plt.vlines(x, 0, 1, label=label)

import pypmc
from pypmc.density.mixture import create_gaussian_mixture, create_t_mixture
from pypmc.mix_adapt.variational import GaussianInference
from pypmc.sampler.importance_sampling import ImportanceSampler

class MultiGaussNaiveBayesClassifier(object):
	def __init__(self, all_labels=None, nmin_multigauss=10, Ngauss_init=5, 
		VB_iter=1000, noteworthy_information = 100, 
		plot_prefix = None, column_names = None):
		self.all_labels = all_labels
		self.nmin_multigauss = nmin_multigauss
		self.Ngauss_init = Ngauss_init
		self.VB_iter = VB_iter
		self.minprob = 1e-5
		self.noteworthy_information = noteworthy_information
		self.plot_prefix = plot_prefix
		self.column_names = column_names
		self.islogvar = None
		pass
	def _make_mixture(x):
		feature_distributions = []
		for i in range(x.shape[1]):
			col = x[:,i]
			col = col[numpy.isfinite(col)]
			std = col.std()
			if len(std) == 0 or not(std > 0): # no useful samples
			else:
				mix = None
				mix = create_gaussian_mixture(numpy.array([[col.mean()]]), [numpy.array([[std]])])
			elif len(col) >= self.nmin_multigauss:
				lo, hi = col.min(), col.max()
				means = numpy.linspace(lo, hi, self.Ngauss_init).reshape((-1,1))
				covs = [numpy.array([[hi - lo]])] * self.Ngauss_init
				mix = create_gaussian_mixture(means, covs)
				
				if self.VB_iter > 0:
					vb = GaussianInference(col.reshape(-1,1), 
						initial_guess=mix, W0=numpy.eye(1)*1e10)
					vb_prune = 0.5 * len(vb.data) / vb.K
					print('running variational Bayes ...')
					vb.run(self.VB_iter, rel_tol=1e-8, abs_tol=1e-5, 
						prune=vb_prune, verbose=True)
					print('running variational Bayes ... done')
					mix = vb.make_mixture()
			feature_distributions.append(mix)
		return feature_distributions
	
	def evaluate_feature_dist(self, feature_distributions, x, logprior):
		assert len(feature_distributions) == x.shape[1]
		logprob = numpy.zeros((len(x), len(feature_distributions)))
		for i, mix in enumerate(feature_distributions)
			if mix is None:
				pass
			else:
				means = numpy.array([g.mu for g in mix.components])
				stdevs = numpy.array([g.sigma[0] for g in mix.components])
				weights = mix.weights.reshape(means.shape)
				
				logprob_col = - 0.5 * log(stdevs**2 * 2 * pi) \
					- 0.5 * ((x[:,i].reshape((1,-1)) - means) / stdevs)**2 \
					+ log(weights)
				
				# sum across gaussian components:
				logprob_col = scipy.misc.logsumexp(logprob_col, axis=0)
				assert len(logprob_col) == len(x), (logprob_col.shape, x.shape)
				logprob[:,i] = logprob_col

		# handle missing values:
		logprob[~numpy.isfinite(logprob)] = 0
		# avoid overly small values:
		logprob[logprob < log(self.minprob)] = log(1e-5)
		# multiply across columns:
		logprob = logprob.sum(axis=1)
		# combine with prior
		return logprob + logprior
		
			
	def compute_gaussian(means, stdevs, x, logprior = 0):
		means = means.reshape((1, -1))
		stdevs = stdevs.reshape((1, -1))
		# combine with prior and return linear value
		return exp(logprob + logprior)
	
	def evaluate_information_single(self, columns, XT, l1, features1, l2, features2):
		lasti = None
		made_plot = False
		if columns is None:
			columns = ['col%d' % (i+1) for i in range(len(XT.shape[1]))]
		
		doplot = self.plot_prefix is not None
		Nsamples = 2000
		for i, (col, mix1, mix2) in enumerate(zip(columns, features1, features2)):
			if mix1 is None or mix2 is None: 
				# one of the two features does not have a distribution here
				continue

			means1 = numpy.array([g.mu[0] for g in mix1.components])
			stdevs1 = numpy.array([g.sigma[0][0] for g in mix1.components])
			weights1 = mix1.weights
			means2 = numpy.array([g.mu for g in mix2.components])
			stdevs2 = numpy.array([g.sigma[0] for g in mix2.components])
			weights2 = mix2.weights

			sampler = ImportanceSampler(mix2.evaluate, mix1)
			sampler.run(Nsamples)
			KL = log(sampler.weights+1e-300).mean()
			print('%3d %3d | %-20s%s | %6d nat   %s' % (l1, l2, col, 
				' log' if islogvar[0,i] else '    ', KL, '***' if KL > self.noteworthy_information else ''))
			if KL > self.noteworthy_information and doplot:
				if lasti is not None :
					mask1 = Y == l1
					plt.plot(XT[mask1,i], XT[mask1,lasti], 's ', ms=2, label='class %d: %d' % (l1, mask1.sum()))
					mask2 = Y == l2
					plt.plot(XT[mask2,i], XT[mask2,lasti], 'o ', ms=2, label='class %d: %d' % (l2, mask2.sum()))
					mask = ~numpy.logical_or(mask1, mask2)
					plt.plot(XT[mask,i], XT[mask,lasti], 'x', ms=2, label='other', color='gray', alpha=0.5)
					plt.xlabel(columns[i])
					plt.ylabel(columns[lasti])
					if islogvar[0,i]:
						plt.xscale('log')
					if islogvar[0,lasti]:
						plt.yscale('log')
					plt.legend(loc='best')
					plt.savefig(self.plot_prefix + '%dvs%d_%s-%s.pdf' % (l1, l2, columns[i], columns[lasti]))
					plt.close()
					made_plot = True
				lasti = i
				
				mask1 = Y == l1
				plothist(XT[mask1,lasti], label='class %d: %d' % (l1, mask1.sum()))
				mask2 = Y == l2
				plothist(XT[mask2,lasti], label='class %d: %d' % (l2, mask2.sum()))
				mask = ~numpy.logical_or(mask1, mask2)
				plothist(XT[mask,lasti], label='other')
				lo, hi = numpy.nanmin(XT), numpy.nanmax(XT)
				samples = numpy.linspace(lo, hi, 400)
				y1 = numpy.sum([scipy.stats.norm(m, s).cdf(samples) * w for m, s, w in zip(means1, stdevs1, weights1), axis=0)
				y2 = numpy.sum([scipy.stats.norm(m, s).cdf(samples) * w for m, s, w in zip(means2, stdevs2, weights2), axis=0)
				assert len(y1) == len(samples)
				plt.plot(samples, y1, '--')
				plt.plot(samples, y2, ':')
				plt.xlabel(columns[lasti])
				plt.legend(loc='best')
				plt.savefig(self.plot_prefix + '%dvs%d_%s.pdf' % (l1, l2, columns[lasti]))
				plt.close()
	
	def fit(self, X, y):
		if self.islogvar is None:
			self.islogvar = ~(X < 0).any(axis=0).reshape((1, -1))
		
		XT = numpy.where(self.islogvar, log10(X), X)

		if self.all_labels is not None:
			self.all_labels = self.all_labels
		else:
			self.all_labels = y

		class_properties = []
		j = []
		for i in enumerate(self.all_labels):
			mask = Y == label
			if not mask.any():
				Xlogprobs[:,i] = 0
				continue
			
			feature_distributions = self.make_mixture(XT[mask,:])
			class_properties.append((label, feature_distributions))
			
			j.append(i)

		for l1, feature_distributions in class_properties:
			for l2, feature_distributions2 in class_properties:
				if l2 <= l1: continue
				self.evaluate_information_single(self.column_names, XT, l1, feature_distributions, l2, feature_distributions2)
		self.class_properties = class_properties
	
	def transform(self, X):
		Ncovered = len(self.all_labels)
		logprior = -log(Ncovered)
		
		XT = numpy.where(self.islogvar, log10(X), X)
		Xlogprobs = numpy.empty((len(X), len(self.class_properties)))
		for i, (l1, feature_distributions) in enumerate(class_properties):
			Xlogprobs[:,i] = self.evaluate_feature_dist(feature_distributions, XT, logprior)

		Xcomplement = 1 - Xlogprobs.sum(axis=1)
		Xlogprobs /= Xlogprobs.sum(axis=1).reshape((-1,1))
		#Xlogprobs[-1] = Xcomplement
		Xlogprobs





name = 'GaussianNaiveBayes'
all_classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95, 99]
Ncovered = len(labels)
logprior = -log(Ncovered)
print('%d columns, %d classes, %d training samples' % (X.shape[1], Ncovered, len(X)))

def compute_gaussian(means, stdevs, x, logprior = 0):
	means = means.reshape((1, -1))
	stdevs = stdevs.reshape((1, -1))
	logprob = - 0.5 * log(stdevs**2 * 2 * pi) - 0.5 * ((x - means) / stdevs)**2
	# handle missing values:
	logprob[~numpy.isfinite(logprob)] = 0
	# avoid overly small values:
	logprob[logprob < log(1e-5)] = log(1e-5)
	# multiply across columns:
	logprob = logprob.sum(axis=1)
	# combine with prior and return linear value
	return exp(logprob + logprior)

def compute_means_stdevs(x):
	means = numpy.empty(x.shape[1])
	stdevs = numpy.empty(x.shape[1])
	for i in range(x.shape[1]):
		col = x[:,i]
		col = col[numpy.isfinite(col)]
		means[i] = col.mean()
		stdevs[i] = col.std()
	return means, stdevs

t0 = time()
print()
print('running %s ...' % name)
Xlogprobs = numpy.empty((len(X), len(all_classes)))
Xlogprobs_simple = numpy.empty((len(X), Ncovered))
islogvar = ~(X < 0).any(axis=0).reshape((1, -1))
XT = numpy.where(islogvar, log10(X), X)
if execute:
	Ylogprobs = numpy.empty((len(unknown), len(all_classes)))
	Ylogprobs_simple = numpy.empty((len(unknown), Ncovered))
	unknownT = numpy.where(islogvar, log10(unknown), unknown)

class_properties = {}
j = []
for i, label in enumerate(all_classes):
	if label not in labels:
		Xlogprobs[:,i] = 0
		if execute:
			Ylogprobs[:,i] = 0
		continue
	mask = Y == label
	assert mask.any()
	means, stdevs = compute_means_stdevs(XT[mask,:])
	class_properties[label] = (means, stdevs)
	Xlogprobs[:,i] = compute_gaussian(means, stdevs, XT, logprior)
	if execute:
		Ylogprobs[:,i] = compute_gaussian(means, stdevs, unknownT, logprior)
	j.append(i)

for l1, (means1, stdevs1) in class_properties.items():
	for l2, (means2, stdevs2) in class_properties.items():
		if l2 <= l1: continue
		lasti = None
		made_plot = False
		for i, (col, mu1, s1, mu2, s2) in enumerate(zip(columns, means1, stdevs1, means2, stdevs2)):
			if not (s1 > 0 and s2 > 0): continue
			KL = ((mu1 - mu2)**2 + s1**2 - s2**2) / (2 * s2**2) + log(s2/s1)
			print('%3d %3d | %-20s%s | %6d nat   %s' % (l1, l2, col, 
				' log' if islogvar[0,i] else '    ', KL, '***' if KL > 10000 else ''))
			if KL > 10000:
				if lasti is not None:
					mask1 = Y == l1
					plt.plot(X[mask1,i], X[mask1,lasti], 's ', ms=2, label='class %d: %d' % (l1, mask1.sum()))
					mask2 = Y == l2
					plt.plot(X[mask2,i], X[mask2,lasti], 'o ', ms=2, label='class %d: %d' % (l2, mask2.sum()))
					mask = ~numpy.logical_or(mask1, mask2)
					plt.plot(X[mask,i], X[mask,lasti], 'x', ms=2, label='other', color='gray', alpha=0.5)
					plt.xlabel(columns[i])
					plt.ylabel(columns[lasti])
					if islogvar[0,i]:
						plt.xscale('log')
					if islogvar[0,lasti]:
						plt.yscale('log')
					plt.legend(loc='best')
					plt.savefig('viz_NaiveBayes_%dvs%d_%s-%s.pdf' % (l1, l2, columns[i], columns[lasti]))
					plt.close()
					made_plot = True
				lasti = i
		if not made_plot and lasti is not None:
			mask1 = Y == l1
			plothist(XT[mask1,lasti], label='class %d: %d' % (l1, mask1.sum()))
			mask2 = Y == l2
			plothist(XT[mask2,lasti], label='class %d: %d' % (l2, mask2.sum()))
			mask = ~numpy.logical_or(mask1, mask2)
			plothist(XT[mask,lasti], label='other')
			plt.xlabel(columns[lasti])
			plt.legend(loc='best')
			plt.savefig('viz_NaiveBayes_%dvs%d_%s.pdf' % (l1, l2, columns[lasti]))
			plt.close()

j = numpy.array(j)
print("logloss: %.3f" % my_log_loss(Y, Xlogprobs[:,j], eps=1e-15, normalize=True, labels=labels))

if execute:
	Xcomplement = 1 - Xlogprobs.sum(axis=1)
	Xlogprobs /= Xlogprobs.sum(axis=1).reshape((-1,1))
	Xlogprobs[-1] = Xcomplement
	Ycomplement = 1 - Ylogprobs.sum(axis=1)
	Ylogprobs /= Ylogprobs.sum(axis=1).reshape((-1,1))
	Ylogprobs[-1] = Ycomplement
	#Xlogprobs = -1. / log(Xlogprobs / Xlogprobs.max(axis=1).reshape((-1,1))+1e-300)
	#Ylogprobs = -1. / log(Ylogprobs / Ylogprobs.max(axis=1).reshape((-1,1))+1e-300)
	print('predictions for training data...')
	numpy.savetxt(training_data_file + '_predictions_%s.csv.gz' % name, Xlogprobs, delimiter=',', fmt='%.4e')
	print('predictions for unknown data...')
	numpy.savetxt(unknown_data_file + '_predictions_%s.csv.gz' % name, Ylogprobs, delimiter=',', fmt='%.4e')
	print('predictions done after %.1fs' % (time() - t0))

"""
N = int(os.environ.get('NNEURONS', '40'))
t0 = time()
print("training MLP...")
clf = MLPClassifier(hidden_layer_sizes=N, max_iter=2000)
q = cross_val_score(clf, log(Xlogprobs[:,j] + 1e-100), Y, cv=3, scoring=scorer)
print('%.3f +- %.3f' % (q.mean(), q.std()))
print('training done after %.1fs' % (time() - t0))

print('Predicting ...')
t0 = time()
print('  predictions for training data...')
predictions = cross_val_predict(clf, log(Xlogprobs[:,j] + 1e-100), Y, cv=5, method='predict_proba')
print('    saving ...')
numpy.savetxt(training_data_file + '_predictions_%sNN.csv.gz' % name, predictions, delimiter=',', fmt='%.3e')
print('  predictions for unknown data...')
clf.fit(log(Xlogprobs[:,j] + 1e-100), Y)
predictions = clf.predict_proba(log(Ylogprobs[:,j] + 1e-100))
print('    saving ...')
numpy.savetxt(unknown_data_file + '_predictions_%sNN.csv.gz' % name, predictions, delimiter=',', fmt='%.3e')
print('predictions done after %.1fs' % (time() - t0))

"""

