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

def get_fitted_classifiers(lcdataset, method, train_config, load_rootdir,
	max_model_ids=20,
	kf=0,
	):
	classifier_dict = {}
	model_ids = list(range(0, max_model_ids))
	bar = ProgressBar(len(model_ids))
	for id in model_ids:
		brf_kwargs = {
			'n_jobs':C_.N_JOBS,
			'n_estimators':200,
			#'max_depth':20,
			#'max_features':'auto',
			#'class_weight':None,
			#'criterion':'entropy',
			#'min_samples_split':2,
			#'min_samples_leaf':1,
			#'verbose':1,
			'bootstrap':True,
			'max_samples':100, # REALLY IMPORTANT PARAMETER
			#'class_weight':'balanced_subsample',
		}
		brf = BalancedRandomForestClassifier(**brf_kwargs)
		
		### fit
		if train_config=='r':
			r_train_lcset_name = f'{kf}@train'
			class_names = lcdataset[r_train_lcset_name].class_names
			x_df, y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			#x_df = pd.concat([r_x_df]*real_repeat*2, axis=0)
			#y_df = pd.concat([r_y_df]*real_repeat*2, axis=0)

		elif train_config=='s':
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[s_train_lcset_name].class_names
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')
			valid_indexs = list(y_df.loc[y_df['__fullsynth__']==1].index)
			x_df = x_df.loc[valid_indexs]
			y_df = y_df.loc[valid_indexs]
			#x_df = pd.concat([s_x_df]*2, axis=0)
			#y_df = pd.concat([s_y_df]*2, axis=0)

		elif train_config=='rs':
			r_train_lcset_name = f'{kf}@train'
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[r_train_lcset_name].class_names
			r_x_df, r_y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')
			valid_indexs = list(y_df.loc[y_df['__fullsynth__']==1].index)
			x_df = x_df.loc[valid_indexs]
			y_df = y_df.loc[valid_indexs]

			x_to_cat = []
			y_to_cat = []
			unique_indexs, count_indexs = np.unique([valid_index.split('.')[0] for valid_index in valid_indexs], return_counts=True)
			for unique_index,count_index in zip(unique_indexs, count_indexs):
				x_to_cat += [r_x_df.loc[[unique_index]]]*count_index
				y_to_cat += [r_y_df.loc[[unique_index]]]*count_index

			x_df = pd.concat([x_df]+x_to_cat, axis=0)
			y_df = pd.concat([y_df]+y_to_cat, axis=0)

		elif train_config=='r-s':
			r_train_lcset_name = f'{kf}@train'
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[r_train_lcset_name].class_names
			r_x_df, r_y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')
			valid_indexs = list(y_df.loc[y_df['__fullsynth__']==1].index)
			x_df = x_df.loc[valid_indexs]
			y_df = y_df.loc[valid_indexs]

			x_to_cat = []
			y_to_cat = []
			unique_indexs, count_indexs = np.unique([valid_index.split('.')[0] for valid_index in valid_indexs], return_counts=True)
			for unique_index,count_index in zip(unique_indexs, count_indexs):
				x_to_cat += [r_x_df.loc[[unique_index]]]*count_index
				aux_df_y = r_y_df.loc[[unique_index]]
				aux_df_y['__y__'].values[:] = 1
				y_to_cat += [aux_df_y]*count_index
			
			y_df['__y__'].values[:] = 0
			x_df = pd.concat([x_df]+x_to_cat, axis=0)
			y_df = pd.concat([y_df]+y_to_cat, axis=0)

		else:
			raise Exception(f'no train_config: {train_config}')

		bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)}')
		#print(x_df.columns, x_df)
		brf.fit(x_df.values, y_df[['__y__']].values[...,0])

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

def evaluate_classifiers(lcdataset, train_config, lcset_name, classifier_dict, model_ids, load_rootdir):
	lcset = lcdataset[lcset_name]
	class_names = lcset.class_names
	results_dict = {}
	for id in model_ids:
		brf = classifier_dict[id]['brf']
		x_df, y_df = load_features(f'{load_rootdir}/{lcset_name}.ftres')
		y_target = y_df[['__y__']].values[...,0]
		y_pred_p = brf.predict_proba(x_df.values)
		y_pred = np.argmax(y_pred_p, axis=-1)

		results_dict[id] = {
			'lcset_name':lcset_name,
			'class_names':class_names,
			'features':classifier_dict[id]['features'],
			'rank':classifier_dict[id]['rank'],
		}
		if not train_config=='r-s':
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)
			results_dict[id].update({
				'metrics_cdict':metrics_cdict,
				'metrics_dict':metrics_dict,
				'cm':cm,
			})

		else:
			recall = y_pred.mean()
			results_dict[id].update({
				'real-recall':recall,
			})
			print(recall)

	return results_dict