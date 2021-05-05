from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.ranks import TopRank
from flamingchoripan.datascience.metrics import get_multiclass_metrics
import numpy as np
import random

###################################################################################################################################################

def clean_df_nans(df, df_values, nan_value,
	nan_mode='value', # value, mean
	):
	if nan_mode=='value':
		df = df.replace([np.inf, -np.inf], np.nan)
		return df.fillna(nan_value)
	elif nan_mode=='mean':
		return df.fillna(df_values)

def train_classifier(train_df_x, train_df_y,
	nan_mode='value', # value, mean
	):
	brf_kwargs = {
		'n_jobs':C_.N_JOBS,
		'n_estimators':2000, # 1000
		#'max_depth':10, #
		'max_features':None,
		#'max_features':'auto',
		'criterion':'entropy', # entropy gini
		#'min_samples_split':2,
		#'min_samples_leaf':1,
		#'verbose':1,
		'bootstrap':True,
		'max_samples':500, # REALLY IMPORTANT PARAMETER
	}

	brf = BalancedRandomForestClassifier(**brf_kwargs)
	mean_train_df_x = train_df_x.mean(axis='index', skipna=True)
	#with pd.option_context('display.max_rows', None, 'display.max_columns', None):
	#	print('mean_train_df_x',mean_train_df_x)
	null_cols = train_df_x.columns[train_df_x.isnull().all()]
	print('null_cols',null_cols)
	train_df_x = clean_df_nans(train_df_x, mean_train_df_x, C_.NAN_VALUE, nan_mode)
	brf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])
	d = {
		'brf':brf,
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		}
	return d

def evaluate_classifier(brf, eval_df_x, eval_df_y, lcset_info,
	):
	class_names = lcset_info['class_names']
	y_target = eval_df_y[['_y']].values[...,0]
	y_pred_p = brf.predict_proba(eval_df_x.values)
	y_pred = np.argmax(y_pred_p, axis=-1)

	wrongs_indexs = ~(y_target==y_pred)
	wrongs_df = eval_df_y[wrongs_indexs]
	metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

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