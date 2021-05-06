from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir
import os

###################################################################################################################################################

def get_multiband_features(invalid_features,
	band_names=['g', 'r'],
	):
	x = []
	for i in invalid_features:
		for b in band_names:
			x.append(f'{i}_{b}')
	return x

def load_features(filedir,
	mode='all',
	):
	df_xy = pd.read_parquet(os.path.abspath(f'{filedir}')) # parquet
	columns = list(df_xy.columns)
	y_columns = ['_y', '_fullsynth']
	df_y = df_xy[y_columns]
	df_x = df_xy[[c for c in columns if not c in y_columns]]

	if mode=='all':
		return df_x, df_y

	elif mode=='sne':
		invalid_features = []
		query_features = get_multiband_features(C_.SNE_SELECTED_FEATURES)
		invalid_features = get_multiband_features(invalid_features)
		df_x = df_xy[[c for c in df_x.columns if c in query_features and not c in invalid_features]]
		return df_x, df_y

def save_features(df_x, df_y, save_filedir):
	save_rootdir = '/'.join([s for s in save_filedir.split('/')[:-1]])
	create_dir(save_rootdir)
	df_xy =  pd.concat([df_y, df_x], axis=1)
	df_xy.to_parquet(os.path.abspath(f'{save_filedir}')) # parquet