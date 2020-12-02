from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
from sklearn.preprocessing import QuantileTransformer, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA, KernelPCA, FastICA
from sklearn.manifold import TSNE
from umap import UMAP
from .files import load_features

###################################################################################################################################################

def get_fitted_maps2d(lcdataset, lcset_name, load_rootdir,
	mode='pca+umap',
	random_state=42,
	pre_out_dims=10,
	out_dims=2,

	metric='euclidean', # default: euclidean
	min_dist=0.1, # default: 0.1
	n_neighbors=15, # default: 15
	):
	lcset = lcdataset[lcset_name]

	#map_scaler = QuantileTransformer(n_quantiles=5000, random_state=random_state, output_distribution='normal')
	map_scaler = QuantileTransformer(n_quantiles=5000, random_state=random_state, output_distribution='uniform')
	#map_scaler = StandardScaler()
	#map_scaler = MinMaxScaler()

	#map_pca = FastICA(n_components=2)#, kernel='rbf', gamma=0.1)
	#map_pca = PCA(n_components=3)
	#map_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
	#map_umap = UMAP(n_components=2, **umap_kwargs)
	#map_tsne = TSNE(n_components=2, **tsne_kwargs)

	load_filedir = f'{load_rootdir}/features/{lcset_name}.ftres'
	df_x, df_y = load_features(load_filedir)
	lcobj_names = list(df_x.index)
	lcobj_names_synth = []
	lcobj_names_real = []
	for lcobj_name in lcobj_names:
		if lcobj_name.split('.')[-1]=='0':
			lcobj_names_real.append(lcobj_name)
		else:
			lcobj_names_synth.append(lcobj_name)

	x = df_x.to_numpy()
	for missing_values in [-C_.NAN_VALUE, C_.NAN_VALUE]:
		x = SimpleImputer(missing_values=missing_values, strategy='median').fit_transform(x)

	x = map_scaler.fit_transform(x)
	y = df_y.to_numpy()[...,0]

	if mode=='umap':
		map_kwargs = {
			'metric':metric,
			'min_dist':min_dist,
			'n_neighbors':n_neighbors,
			'random_state':random_state,
			'transform_seed':random_state,
		}
		map_obj = UMAP(n_components=2, **map_kwargs)
		map_x = map_obj.fit_transform(x, y=y)

	elif mode=='pca+umap':
		map_kwargs = {
			'metric':metric,
			'min_dist':min_dist,
			'n_neighbors':int(n_neighbors),
			'random_state':random_state,
			'transform_seed':random_state,
		}
		pca = PCA(n_components=pre_out_dims)
		map_obj = UMAP(n_components=out_dims, **map_kwargs)
		#method_name = '$\\text{PCA}_{'+str(pre_out_dims)+'}\\to\\text{UMAP}_{'+str(out_dims)+'}$'
		method_name = '$PCA_{'+str(pre_out_dims)+'} + UMAP_{'+str(out_dims)+'}$'

		map_x = map_obj.fit_transform(pca.fit_transform(x), y=y)

	elif mode=='tsne':
		map_kwargs = {
			'perplexity':50.0, # default: 30
			'random_state':random_state,
		}

	else:
		raise Exception(f'no mode {mode}')
	#map_pca.fit(x) # fit
	#
	#map_tsne.fit(x) # fit

	d = {
		'method_name':method_name,
		'lcset_name':lcset_name,
		'scaler':map_scaler,
		'map_obj':map_obj,
		'map_x':map_x,
		'y':y,
		'class_names':lcset.class_names,
		'lcobj_names':lcobj_names,
		'lcobj_names_synth':lcobj_names_synth,
		'lcobj_names_real':lcobj_names_real,
	}
	return d