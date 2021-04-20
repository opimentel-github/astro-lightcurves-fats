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

def train_classifier(train_df_x, train_df_y,
	):
	brf_kwargs = {
		'n_jobs':C_.N_JOBS,
		'n_estimators':1000, # 1000
		#'max_depth':10, # #
		'max_features':None,
		#'max_features':'auto',
		#'class_weight':None,
		'criterion':'entropy',
		#'min_samples_split':2,
		#'min_samples_leaf':1,
		#'verbose':1,
		'bootstrap':True,
		'max_samples':500, # REALLY IMPORTANT PARAMETER
		#'class_weight':'balanced_subsample',
	}
	brf = BalancedRandomForestClassifier(**brf_kwargs)
	brf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])
	return brf


def evaluate_classifier(brf, eval_df_x, eval_df_y, lcset_info,
	):
	class_names = lcset_info['class_names']
	y_target = eval_df_y[['_y']].values[...,0]
	y_pred_p = brf.predict_proba(eval_df_x.values)
	y_pred = np.argmax(y_pred_p, axis=-1)

	metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

	### results
	features = list(eval_df_x.columns)
	rank = TopRank('features')
	rank.add_list(features, brf.feature_importances_)
	rank.calcule()
	results = {
		'lcset_info':lcset_info,
		'metrics_cdict':metrics_cdict,
		'metrics_dict':metrics_dict,
		'cm':cm,
		'features':features,
		'rank':rank,
	}
	return results