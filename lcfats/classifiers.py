from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
from imblearn.over_sampling import RandomOverSampler, SMOTE
from fuzzytools.datascience.ranks import TopRank
from fuzzytools.datascience.metrics import get_multiclass_metrics
from fuzzytools.dataframes import clean_df_nans
import numpy as np
import random
from fuzzytools.dataframes import DFBuilder
from nested_dict import nested_dict

NAN_MODE = 'value' # value, mean
N_JOBS = C_.N_JOBS

###################################################################################################################################################

def train_classifier(_train_df_x, train_df_y, _val_df_x, val_df_y, lcset_info,
	nan_mode=NAN_MODE,
	):
	class_names = lcset_info['class_names']
	train_df_x, mean_train_df_x, null_cols = clean_df_nans(_train_df_x, mode=NAN_MODE)
	random_sampler = RandomOverSampler( # RandomOverSampler SMOTE
		sampling_strategy='all', # not minority all
		)
	x_rs, y_rs = random_sampler.fit_resample(train_df_x.values, train_df_y[['_y']].values[...,0])
	best_rf = (None, -np.inf)
	for max_depth in [1, 2, 3, 4, 5]:
		for min_samples_split in [2, 3, 4, 5]:
			rf = RandomForestClassifier(
				max_features='auto', # None auto
				max_depth=max_depth,
				n_jobs=N_JOBS,
				class_weight=None,
				criterion='entropy',
				min_samples_split=min_samples_split,
				min_samples_leaf=1,
				n_estimators=1000, # 100 500 1000
				bootstrap=True,
				#verbose=1,
				)
			rf.fit(x_rs, y_rs)
			val_df_x, _, _ = clean_df_nans(_val_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
			y_pred_p = rf.predict_proba(val_df_x.values)
			y_target = val_df_y[['_y']].values[...,0]
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred_p, y_target, class_names)
			rf_metric = metrics_dict['b-f1score'] # recall f1score
			if rf_metric>best_rf[-1]:
				best_rf = (rf, rf_metric)
	
	features = list(train_df_x.columns)
	rank = TopRank('features')
	rank.add_list(features, rf.feature_importances_)
	rank.calcule()
	d = {
		'rf':best_rf[0],
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		'features':features,
		'rank':rank,
		}
	return d

###################################################################################################################################################

def evaluate_classifier(brf_d, eval_df_x, eval_df_y, lcset_info,
	nan_mode=NAN_MODE,
	):
	rf = brf_d['rf']
	mean_train_df_x = brf_d['mean_train_df_x']
	class_names = lcset_info['class_names']
	y_target = eval_df_y[['_y']].values[...,0]
	eval_df_x, _, _ = clean_df_nans(eval_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
	y_pred_p = rf.predict_proba(eval_df_x.values)
	metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred_p, y_target, class_names)

	### wrong samples
	y_pred = np.argmax(y_pred_p, axis=-1)
	lcobj_names = list(eval_df_y.index)
	wrong_classification = ~(y_target==y_pred)
	assert len(lcobj_names)==len(wrong_classification)
	wrongs_df = DFBuilder()
	for kwc,wc in enumerate(wrong_classification):
		if wc:
			wrongs_df.append(lcobj_names[kwc], {'y_target':class_names[y_target[kwc]], 'y_pred':class_names[y_pred[kwc]]})

	### results
	d = {
		'wrongs_df':wrongs_df.get_df(),
		'lcset_info':lcset_info,
		'metrics_cdict':metrics_cdict,
		'metrics_dict':metrics_dict,
		'cm':cm,
		'features':brf_d['features'],
		'rank':brf_d['rank'],
		}
	return d