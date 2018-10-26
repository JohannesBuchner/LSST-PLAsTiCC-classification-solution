from __future__ import print_function, division
from alltrain import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

del train

X = mytransformer.fit_transform(X)

execute = unknown_data_file is not None
unknown = None
if execute:
	unknown = pandas.read_csv(unknown_data_file)
	unknown.pop('object_id')        
	unknown = unknown.values
	print('unknown:', unknown.shape)
	if simplify_space:
		unknown = unknown[:,column_mask]
	unknown = imp.transform(unknown)
	unknown = mytransformer.transform(unknown)

def make_transform(name, tsne, X, Y, Z):
	print()
	print("inputs:", X.shape, Y.shape, numpy.shape(Z))
	sys.stdout.write('running %s ...\r' % name)
	sys.stdout.flush()
	t0 = time()
	# if Z is given, we should merge together X+Z and transform with them
	# then color only the labelled (X,Y) values
	if Z is None:
		XZ = X
		YY = Y
	else:
		XZ = numpy.vstack((X, Z))
		YY = numpy.hstack((Y, numpy.zeros(len(Z))-1 ))
	L = tsne.fit_transform(XZ)
	print('running %s complete (%.2fs)' % (name, time() - t0))
	# plot results
	plt.figure(figsize=(10,10))
	# give each class a different color
	labels_here = numpy.unique(Y)
	for cls in labels_here:
		print("  plotting class %d" % cls)
		mask_cls = YY == cls
		plt.scatter(L[mask_cls,0], L[mask_cls,1], label='class %d' % cls)
	
	plt.title(name)
	mask_cls = YY == -1
	plt.scatter(L[mask_cls,0], L[mask_cls,1], c='gray', marker='x', s=2, label='unknown')
	print("saving plot...")
	plt.legend(loc='best', prop=dict(size=8))
	plt.savefig('viz_%s.pdf' % name)
	plt.close()


n_components = int(os.environ.get('NPCACOMP', '40'))
perplexity = int(os.environ.get('PERPLEXITY', '30'))

if n_components > 0:
	prefix = transform + '-PCA%d-' % n_components
	print("dimensionality reduction with %s ..." % prefix)
	t0 = time()
	pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
	X_white = pca.fit_transform(X)
	print("done in %0.3fs" % (time() - t0))
	print('PCA Variance ratios:', pca.explained_variance_ratio_)
	#X_white = pca.transform(X)
	del X
	if execute:
		unknown_white = pca.transform(unknown)
		del unknown
	else:
		unknown_white = None
else:
	prefix = transform + '-'
	X_white = X
	unknown_white = unknown

tsne = TSNE(perplexity=perplexity, n_iter=5000)
make_transform(prefix + 'TSNE%d' % perplexity, tsne, X_white, Y, unknown_white)




