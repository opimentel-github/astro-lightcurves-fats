from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir
import os

###################################################################################################################################################

def load_features(filedir):
	df_xy = pd.read_parquet(os.path.abspath(f'{filedir}')) # parquet
	columns = list(df_xy.columns)
	y_columns = ['_y', '_fullsynth']
	df_y = df_xy[y_columns]
	df_x = df_xy[[c for c in columns if not c in y_columns]]

	inv = [
		#'IAR_phi', # ???
		#'SF_ML_amplitude', # ?
		#'LinearTrend',

		#'MHPS_low',
		'MHPS_non_zero',
		'MHPS_PN_flag',
		'MHPS_ratio',
		'MHPS_high',

		#'SPM_t0', # important
		'SPM_A', # flux wise
		'SPM_chi', # conflictive

		'pre_peak_LinearTrend',
		'post_peak_LinearTrend',
		'post_peak_LinearTrend1',
		'post_peak_LinearTrend2',
		'post_peak_LinearTrend3',

		#'peak_obs_mu',
		#'peak_obs_std',
		#'peak_days_mu',
		#'peak_days_std',
	]
	inv2 = []
	for i in inv:
		for b in ['g', 'r']:
			inv2.append(f'{i}_{b}')

	df_x = df_xy[[c for c in df_x.columns if not c in inv2]]
	#print(df_y)
	#print(df_x)
	return df_x, df_y

def save_features(df_x, df_y, save_filedir):
	save_rootdir = '/'.join([s for s in save_filedir.split('/')[:-1]])
	create_dir(save_rootdir)
	df_xy =  pd.concat([df_y, df_x], axis=1)
	df_xy.to_parquet(os.path.abspath(f'{save_filedir}')) # parquet