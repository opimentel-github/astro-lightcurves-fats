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

def get_fitted_maps2d(lcdataset, s_lcset_name, load_rootdir,
	mode='pca+umap',
	random_state=0,
	pre_out_dims=10,
	out_dims=2,

	metric='euclidean', # default: euclidean
	min_dist=0.1, # default: 0.1
	n_neighbors=15, # default: 15
	):
	r_lcset_name = s_lcset_name.split('.')[0]
	r_lcset = lcdataset[r_lcset_name]
	s_lcset = lcdataset[s_lcset_name]

	#map_scaler = QuantileTransformer(n_quantiles=5000, random_state=random_state, output_distribution='normal')
	map_scaler = QuantileTransformer(n_quantiles=5000, random_state=random_state, output_distribution='uniform')
	#map_scaler = StandardScaler()
	#map_scaler = MinMaxScaler()

	#map_pca = FastICA(n_components=2)#, kernel='rbf', gamma=0.1)
	#map_pca = PCA(n_components=3)
	#map_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
	#map_umap = UMAP(n_components=2, **umap_kwargs)
	#map_tsne = TSNE(n_components=2, **tsne_kwargs)

	r_df_x, r_df_y = load_features(f'{load_rootdir}/{r_lcset_name}.ftres')
	r_lcobj_names = list(r_df_x.index)

	s_df_x, s_df_y = load_features(f'{load_rootdir}/{s_lcset_name}.ftres')
	s_lcobj_names = list(s_df_x.index)

	map_lcobj_names = r_lcobj_names+s_lcobj_names
	x = np.concatenate([r_df_x.values, s_df_x.values], axis=0)
	y = np.concatenate([r_df_y.values, s_df_y.values], axis=0)[...,0]

	for missing_values in [-C_.NAN_VALUE, C_.NAN_VALUE]:
		x = SimpleImputer(missing_values=missing_values, strategy='median').fit_transform(x)

	x = map_scaler.fit_transform(x)

	if mode=='pca+umap':
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
		method_name = '$PCA_{'+str(pre_out_dims)+'} + UMAP_{'+str(out_dims)+'}$ projection of FATS features\n'
		method_name += f'metric={metric} - min-dist={min_dist:.3f} - n-neighbors={int(n_neighbors)}'
		map_x = map_obj.fit_transform(pca.fit_transform(x), y=y)
		
	'''
	elif mode=='umap':
		map_kwargs = {
			'metric':metric,
			'min_dist':min_dist,
			'n_neighbors':n_neighbors,
			'random_state':random_state,
			'transform_seed':random_state,
		}
		map_obj = UMAP(n_components=2, **map_kwargs)
		map_x = map_obj.fit_transform(x, y=y)




	elif mode=='tsne':
		map_kwargs = {
			'perplexity':50.0, # default: 30
			'random_state':random_state,
		}

	else:
		raise Exception(f'no mode {mode}')
	'''

	d = {
		'method_name':method_name,
		'scaler':map_scaler,
		'map_obj':map_obj,
		'map_lcobj_names':map_lcobj_names,
		'map_x':map_x,
		'y':y,
		'class_names':r_lcset.class_names,
		'r_lcset_name':r_lcset_name,
		's_lcset_name':s_lcset_name,
		'r_lcobj_names':r_lcobj_names,
		's_lcobj_names':s_lcobj_names,
	}
	return d