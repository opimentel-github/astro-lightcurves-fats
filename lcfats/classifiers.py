from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import random
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
from fuzzytools.dataframes import DFBuilder
from fuzzytools.dicts import update_dicts

NAN_MODE = 'value' # value, mean
N_JOBS = C_.N_JOBS

###################################################################################################################################################

def train_classifier(_train_df_x, train_df_y, _val_df_x, val_df_y, lcset_info,
	nan_mode=NAN_MODE,
	):
	class_names = lcset_info['class_names']
	train_df_x, mean_train_df_x, null_cols = clean_df_nans(_train_df_x, mode=NAN_MODE)
	best_rf = None
	best_rf_metric = -np.inf
	for criterion in ['gini', 'entropy']:
		for max_depth in [1, 2, 4, 8, 16][::-1]:
			for max_samples in np.linspace(.1, .9, 6):
			# for max_samples in [None]:
				rf = BalancedRandomForestClassifier( # BalancedRandomForestClassifier RandomForestClassifier
					n_jobs=N_JOBS,
					criterion=criterion,
					max_depth=max_depth,
					n_estimators=1024, # 10 256, 512, 1024, 2048
					max_samples=max_samples,
					max_features='auto', # None auto
					# min_samples_split=min_samples_split,
					bootstrap=True,
					#verbose=1,
					)
				rf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])
				val_df_x, _, _ = clean_df_nans(_val_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
				y_pred_p = rf.predict_proba(val_df_x.values)
				y_true = val_df_y[['_y']].values[...,0]
				metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred_p, y_true, class_names)
				rf_metric = metrics_dict['b-f1score'] # recall f1score
				print(f'samples={len(train_df_y)}; criterion={criterion}; max_depth={max_depth}; max_samples={max_samples}; rf_metric={rf_metric}; best_rf_metric={best_rf_metric}')
				if rf_metric>best_rf_metric:
					best_rf = rf
					best_rf_metric = rf_metric
	
	### save best
	features = list(train_df_x.columns)
	rank = TopRank('features')
	rank.add_list(features, best_rf.feature_importances_)
	rank.calcule()
	print(rank)
	d = {
		'rf':best_rf,
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		'features':features,
		'rank':rank,
		}
	return d

###################################################################################################################################################

def evaluate_classifier(rf_d, eval_df_x, eval_df_y, lcset_info,
	nan_mode=NAN_MODE,
	):
	class_names = lcset_info['class_names']
	features = rf_d['features']

	thdays_class_metrics_df = DFBuilder()
	thdays_class_metrics_cdf = {c:DFBuilder() for c in class_names}
	thdays_predictions = {}
	thdays_cm = {}

	thdays = [100] # fixme
	for thday in thdays:
		rf = rf_d['rf']
		mean_train_df_x = rf_d['mean_train_df_x']
		y_true = eval_df_y[['_y']].values[...,0]
		eval_df_x, _, _ = clean_df_nans(eval_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
		y_pred_p = rf.predict_proba(eval_df_x.values)
		thdays_predictions[thday] = {'y_true':y_true, 'y_pred_p':y_pred_p}
		metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred_p, y_true, class_names)
		for c in class_names:
			thdays_class_metrics_cdf[c].append(thday, update_dicts([{'_thday':thday}, metrics_cdict[c]]))
		thdays_class_metrics_df.append(thday, update_dicts([{'_thday':thday}, metrics_dict]))
		thdays_cm[thday] = cm

		### progress bar
		bmetrics_dict = {k:metrics_dict[k] for k in metrics_dict.keys() if 'b-' in k}
		print(f'bmetrics_dict={bmetrics_dict}')

	d = {
		'model_name':f'mdl=brf',
		'survey':lcset_info['survey'],
		'band_names':lcset_info['band_names'],
		'class_names':class_names,
		'lcobj_names':list(eval_df_y.index),

		'thdays':thdays,
		'thdays_predictions':thdays_predictions,
		'thdays_class_metrics_df':thdays_class_metrics_df.get_df(),
		'thdays_class_metrics_cdf':{c:thdays_class_metrics_cdf[c].get_df() for c in class_names},
		'thdays_cm':thdays_cm,

		'features':features,
		'rank':rf_d['rank'],
		}

	return d