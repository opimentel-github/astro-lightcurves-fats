from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.ranks import TopRank
from flamingchoripan.datascience.metrics import get_multiclass_metrics
from flamingchoripan.dataframes import clean_df_nans
import numpy as np
import random

###################################################################################################################################################

def train_classifier(train_df_x, train_df_y,
	nan_mode='value', # value, mean
	):
	min_population_samples = min(np.unique(train_df_y['_y'].values, return_counts=True)[-1])
	brf_kwargs = { # same as ALERCE
		'max_features':'auto', # None auto
		'max_depth':None,
		'n_jobs':C_.N_JOBS,
		'class_weight':None,
		'criterion':'entropy',
		'min_samples_split':2,
		'min_samples_leaf':1,

		'n_estimators':10000, # 500 1000 2000
		#'sampling_strategy':'not minority',
		'sampling_strategy':'all',
		'bootstrap':True,
		'replacement':True,
		'max_samples':100, # *** # 100 500 1000 min_population_samples
		#'verbose':1,
	}
	brf = BalancedRandomForestClassifier(**brf_kwargs)
	train_df_x, mean_train_df_x, null_cols = clean_df_nans(train_df_x, mode=nan_mode)
	#print('null_cols',null_cols)
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#	print('mean_train_df_x',mean_train_df_x)
	brf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])
	d = {
		'brf':brf,
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		}
	return d

def evaluate_classifier(brf_d, eval_df_x, eval_df_y, lcset_info,
	nan_mode='value', # value, mean
	):
	brf = brf_d['brf']
	mean_train_df_x = brf_d['mean_train_df_x']
	class_names = lcset_info['class_names']
	y_target = eval_df_y[['_y']].values[...,0]
	eval_df_x, _, _ = clean_df_nans(eval_df_x, mode=nan_mode, df_values=mean_train_df_x)
	y_pred_p = brf.predict_proba(eval_df_x.values)

	y_pred = np.argmax(y_pred_p, axis=-1)
	wrongs_indexs = ~(y_target==y_pred)
	wrongs_df = eval_df_y[wrongs_indexs]
	metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred_p, y_target, class_names)

	### results
	features = list(eval_df_x.columns)
	rank = TopRank('features')
	rank.add_list(features, brf.feature_importances_)
	rank.calcule()
	d = {
		'wrongs_df':wrongs_df,
		'lcset_info':lcset_info,
		'metrics_cdict':metrics_cdict,
		'metrics_dict':metrics_dict,
		'cm':cm,
		'features':features,
		'rank':rank,
		}
	return d