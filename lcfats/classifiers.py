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

NAN_MODE = 'value' # value, mean

###################################################################################################################################################

def train_classifier(train_df_x, train_df_y,
	nan_mode=NAN_MODE,
	):
	train_df_x, mean_train_df_x, null_cols = clean_df_nans(train_df_x, mode=NAN_MODE)
	#min_population_samples = min(np.unique(train_df_y['_y'].values, return_counts=True)[-1])
	rf_kwargs = {
		'max_features':'auto', # None auto
		'max_depth':3,
		'n_jobs':C_.N_JOBS,
		'class_weight':None,
		'criterion':'entropy',
		'min_samples_split':2,
		'min_samples_leaf':1,

		'n_estimators':1000, # 500 1000 2000 5000
		'bootstrap':True,
		#'max_samples':10, # *** # 100 500 1000 min_population_samples
		#'verbose':1,
	}
	sampling_strategy = 'all' # not minority all
	rf = RandomForestClassifier(**rf_kwargs)
	random_sampler = RandomOverSampler(sampling_strategy=sampling_strategy) # RandomOverSampler SMOTE
	x_rs, y_rs = random_sampler.fit_resample(train_df_x.values, train_df_y[['_y']].values[...,0])
	#brf = make_pipeline_imb(random_sampler, rf)
	#param_grid = {
	#	'max_depth':[1,2,5,10,15,20],
	#	}
	#grid_clf = GridSearchCV(rf, param_grid, cv=5)
	#grid_clf.fit(x_rs, y_rs)
	#rf = grid_clf.best_estimator_
	
	rf.fit(x_rs, y_rs)
	
	#brf.fit(train_df_x.values, train_df_y[['_y']].values[...,0])

	features = list(train_df_x.columns)
	rank = TopRank('features')
	rank.add_list(features, rf.feature_importances_)
	rank.calcule()
	d = {
		'brf':rf,
		'mean_train_df_x':mean_train_df_x,
		'null_cols':null_cols,
		'features':features,
		'rank':rank,
		}
	return d

def evaluate_classifier(brf_d, eval_df_x, eval_df_y, lcset_info,
	nan_mode=NAN_MODE,
	):
	brf = brf_d['brf']
	mean_train_df_x = brf_d['mean_train_df_x']
	class_names = lcset_info['class_names']
	y_target = eval_df_y[['_y']].values[...,0]
	eval_df_x, _, _ = clean_df_nans(eval_df_x, mode=NAN_MODE, df_values=mean_train_df_x)
	y_pred_p = brf.predict_proba(eval_df_x.values)
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