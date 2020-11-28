from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.statistics import TopRank
from flamingchoripan.datascience.metrics import get_all_metrics_c
from flamingchoripan.progress_bars import ProgressBar
from .files import load_features

###################################################################################################################################################

def get_fitted_classifiers(lcdataset, train_lcset_name, load_rootdir,
	max_model_ids=40,
	remove_original_samples=False,
	):
	train_lcset = lcdataset[train_lcset_name]
	class_names = train_lcset.class_names
	class_brfc_weights_cdict = train_lcset.get_class_brfc_weights_cdict()
	classifier_dict = {}
	model_ids = list(range(0, max_model_ids))
	bar = ProgressBar(len(model_ids))
	for id in model_ids:
		bar(f'training id: {id}')
		brf_kwargs = {
			'n_jobs':C_.N_JOBS,
			'n_estimators':200,
            #'max_features':'auto',
            #'max_depth':None,
			#'class_weight':{kc:class_brfc_weights_cdict[c] for kc,c in enumerate(class_names)},
			#'random_state':0,
			'class_weight':None,
			'criterion':'entropy',
			#'min_samples_split':2,
            #'min_samples_leaf':1,
			#'verbose':1,
		}
		### fit
		brf = BalancedRandomForestClassifier(**brf_kwargs)
		#brf = RandomForestClassifier(**brf_kwargs)
		x_df, y_df = load_features(f'{load_rootdir}/{train_lcset_name}.ftres')

		if remove_original_samples:
			to_drop = [i for i in list(x_df.index) if ('.' in i and i.split('.')[-1]=='0')]
			x_df = x_df.drop(to_drop)
			y_df = y_df.drop(to_drop)

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
		scores_cdict, scores_dict, cm = get_all_metrics_c(y_pred, y_target, class_names, pred_is_onehot=False)
		results_dict[id] = {
			'lcset_name':lcset_name,
			'class_names':class_names,
			'cm':cm,
			'f1score':scores_dict['f1score'],
		}

	return results_dict