from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.statistics import TopRank
from flamingchoripan.datascience.metrics import get_multiclass_metrics
from flamingchoripan.progress_bars import ProgressBar
from .files import load_features
import numpy as np

###################################################################################################################################################

def get_fitted_classifiers(lcdataset, train_lcset_name, load_rootdir,
	max_model_ids=20,
	add_real_samples=False,
	real_repeat=16,
	):
	train_lcset = lcdataset[train_lcset_name]
	class_names = train_lcset.class_names
	classifier_dict = {}
	model_ids = list(range(0, max_model_ids))
	bar = ProgressBar(len(model_ids))
	for id in model_ids:
		brf_kwargs = {
			'n_jobs':C_.N_JOBS,
			'n_estimators':50,
			#'max_depth':20,
			#'max_features':'auto',
			#'class_weight':None,
			#'criterion':'entropy',
			#'min_samples_split':2,
			#'min_samples_leaf':1,
			#'verbose':1,
			'max_samples':100, # REALLY IMPORTANT PARAMETER
		}
		### fit
		brf = BalancedRandomForestClassifier(**brf_kwargs)
		#brf = RandomForestClassifier(**brf_kwargs)
		x_df, y_df = load_features(f'{load_rootdir}/{train_lcset_name}.ftres')

		if add_real_samples:
			real_lcset_name = train_lcset_name.split('.')[0]
			#print(real_lcset_name)
			rx_df, ry_df = load_features(f'{load_rootdir}/{real_lcset_name}.ftres')
			#rx_df = rx_df
			#ry_df = ry_df
			#x_df = pd.concat([rx_df]*real_repeat, axis=0, ignore_index=True)
			#y_df = pd.concat([ry_df]*real_repeat, axis=0, ignore_index=True)
			x_df = pd.concat([x_df]+[rx_df]*real_repeat, axis=0)
			y_df = pd.concat([y_df]+[ry_df]*real_repeat, axis=0)

		bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)}')
		#print(x_df.columns, x_df)
		brf.fit(x_df.values, y_df.values[...,0])

		### rank
		features = list(x_df.columns)
		rank = TopRank('features')
		rank.add_list(features, brf.feature_importances_)
		rank.calcule_rank()
		classifier_dict[id] = {
			'brf':brf,
			'features':features,
			'rank':rank,
		}
	bar.done()
	return classifier_dict, model_ids

def evaluate_classifiers(lcdataset, lcset_name, classifier_dict, model_ids, load_rootdir):
	lcset = lcdataset[lcset_name]
	class_names = lcset.class_names
	results_dict = {}
	for id in model_ids:
		brf = classifier_dict[id]['brf']
		x_df, y_df = load_features(f'{load_rootdir}/{lcset_name}.ftres')
		y_target = y_df.values[...,0]
		y_pred = brf.predict(x_df.values)
		metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False)
		results_dict[id] = {
			'lcset_name':lcset_name,
			'class_names':class_names,
			'metrics_cdict':metrics_cdict,
			'metrics_dict':metrics_dict,
			'cm':cm,
			'features':classifier_dict[id]['features'],
			'rank':classifier_dict[id]['rank'],
		}

	return results_dict