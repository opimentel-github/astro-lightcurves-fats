from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.statistics import TopRank
from flamingchoripan.datascience.metrics import get_all_metrics_c

###################################################################################################################################################

def get_fitted_classifiers(lcdataset, train_lcset_name,
	max_model_ids=20,
	):
	train_lcset = lcdataset[train_lcset_name]
	root_folder = f'../save/{train_lcset.survey}'
	classifier_dict = {}
	model_ids = list(range(0, max_model_ids))
	for id in model_ids:
		brf_kwargs = {
			'max_depth':50,
			#'weights':train_lcset.get_class_weights(),
			#'random_state':0,
			'n_jobs':C_.N_JOBS,
			#'verbose':1,
		}
		### fit
		brf = BalancedRandomForestClassifier(**brf_kwargs)
		#brf = RandomForestClassifier(**brf_kwargs)
		x_df = pd.read_parquet(f'{root_folder}/{train_lcset_name}.x.parquet')
		y_df = pd.read_parquet(f'{root_folder}/{train_lcset_name}.y.parquet')
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

	return classifier_dict, model_ids

def evaluate_classifiers(lcdataset, lcset_name, classifier_dict, model_ids):
	lcset = lcdataset[lcset_name]
	class_names = lcset.class_names
	root_folder = f'../save/{lcset.survey}'
	results_dict = {}
	for id in model_ids:
		brf = classifier_dict[id]['brf']
		x_df = pd.read_parquet(f'{root_folder}/{lcset_name}.x.parquet')
		y_df = pd.read_parquet(f'{root_folder}/{lcset_name}.y.parquet')
		y_target = y_df.values[...,0]
		y_pred = brf.predict(x_df.values)
		scores_cdict, scores_dict, cm = get_all_metrics_c(y_pred, y_target, class_names, pred_is_onehot=False)
		results_dict[id] = {
			'lcset_name':lcset_name,
			'class_names':class_names,
			'cm':cm,
			'f1score':scores_dict['f1score'],
		}

	return results_dict