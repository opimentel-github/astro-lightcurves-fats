from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from flamingchoripan.datascience.statistics import TopRank
from flamingchoripan.datascience.metrics import get_multiclass_metrics
from flamingchoripan.progress_bars import ProgressBar
import numpy as np
import random

###################################################################################################################################################

def train_classifiers(lcdataset, method, train_config, test_lcset_name, load_rootdir,
	max_model_ids=20,
	kf=0,
	):
	results_dict = {}
	model_ids = list(range(0, max_model_ids))
	bar = ProgressBar(len(model_ids))
	for id in model_ids:
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
		
		### fit
		if train_config=='r':
			class_names = lcdataset[f'{kf}@train'].class_names
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@train.ftres')

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			brf.fit(x_df.values, y_df[['__y__']].values[...,0])

			### evaluate
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@{test_lcset_name}.ftres')
			y_target = y_df[['__y__']].values[...,0]
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			#print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				'metrics_cdict':metrics_cdict,
				'metrics_dict':metrics_dict,
				'cm':cm,
			}

		elif train_config=='s':
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[s_train_lcset_name].class_names
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			brf.fit(x_df.values, y_df[['__y__']].values[...,0])

			### evaluate
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@{test_lcset_name}.ftres')
			y_target = y_df[['__y__']].values[...,0]
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			#print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				'metrics_cdict':metrics_cdict,
				'metrics_dict':metrics_dict,
				'cm':cm,
			}

		elif train_config=='r+s':
			r_train_lcset_name = f'{kf}@train'
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[r_train_lcset_name].class_names
			r_x_df, r_y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			s_x_df, s_y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')

			x_df = pd.concat([r_x_df, s_x_df], axis=0)
			y_df = pd.concat([r_y_df, s_y_df], axis=0)

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			brf.fit(x_df.values, y_df[['__y__']].values[...,0])

			### evaluate
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@{test_lcset_name}.ftres')
			y_target = y_df[['__y__']].values[...,0]
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			#print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				'metrics_cdict':metrics_cdict,
				'metrics_dict':metrics_dict,
				'cm':cm,
			}

		elif train_config=='r2+s2':
			r_train_lcset_name = f'{kf}@train'
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[r_train_lcset_name].class_names
			r_x_df, r_y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')
			valid_indexs = list(y_df.index)
			#valid_indexs = list(y_df.loc[y_df['__fullsynth__']==1].index) # filter by only full synthetic
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

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			brf.fit(x_df.values, y_df[['__y__']].values[...,0])

			### evaluate
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@{test_lcset_name}.ftres')
			y_target = y_df[['__y__']].values[...,0]
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			metrics_cdict, metrics_dict, cm = get_multiclass_metrics(y_pred, y_target, class_names, pred_is_onehot=False, y_pred_p=y_pred_p)

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			#print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				'metrics_cdict':metrics_cdict,
				'metrics_dict':metrics_dict,
				'cm':cm,
			}

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
				y_to_cat += [r_y_df.loc[[unique_index]]]*count_index

			x_df = pd.concat([x_df]+x_to_cat, axis=0)
			y_df = pd.concat([y_df]+y_to_cat, axis=0)

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			#x_df['SPM_beta_r'].values[:] = y_df['__fullsynth__'].values[:] # dummy
			brf.fit(x_df.values, y_df[['__fullsynth__']].values[...,0])

			### evaluate
			x_df, y_df = load_features(f'{load_rootdir}/{kf}@{test_lcset_name}.ftres')
			y_target = y_df[['__fullsynth__']].values[...,0]
			#x_df['SPM_beta_r'].values[:] = y_df['__fullsynth__'].values[:] # dummy
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			
			print(np.mean((y_pred==y_target).astype(np.float)))
			print(np.mean(-np.log(y_pred_p[:,0]+1e-10)))

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			#print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				#'metrics_cdict':metrics_cdict,
				#'metrics_dict':metrics_dict,
				#'cm':cm,
			}

		elif train_config=='0r-s':
			r_train_lcset_name = f'{kf}@train'
			s_train_lcset_name = f'{kf}@train.{method}'
			class_names = lcdataset[r_train_lcset_name].class_names
			r_x_df, r_y_df = load_features(f'{load_rootdir}/{r_train_lcset_name}.ftres')
			x_df, y_df = load_features(f'{load_rootdir}/{s_train_lcset_name}.ftres')
			valid_indexs = list(y_df.loc[y_df['__fullsynth__']==1].index)
			s_x_df = x_df.loc[valid_indexs]
			s_y_df = y_df.loc[valid_indexs]

			real_obj_names = [valid_index.split('.')[0] for valid_index in valid_indexs]
			random.shuffle(real_obj_names)
			x_df, y_df = get_mixed_df(real_obj_names[len(real_obj_names)//4:], r_x_df, r_y_df, s_x_df, s_y_df, replace_class=True)

			### fit
			bar(f'training id: {id} - samples: {len(y_df)} - features: {len(x_df.columns)} {list(x_df.columns)}')
			brf.fit(x_df.values, y_df[['__y__']].values[...,0])

			### evaluate
			x_df, y_df = get_mixed_df(real_obj_names[:len(real_obj_names)//4], r_x_df, r_y_df, s_x_df, s_y_df, replace_class=True)
			y_target = y_df[['__y__']].values[...,0]
			y_pred_p = brf.predict_proba(x_df.values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			print(np.mean(y_target==y_pred).astype(np.float))

			### results
			features = list(x_df.columns)
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				#'real-recall':recall, # low is good
				#'real-xentropy':xentropy, # high is good
			}

		elif train_config=='1r-s':
			r_lcset = lcdataset[f'{kf}@train']
			s_lcset = lcdataset[f'{kf}@train.{method}']

			xy_df = []
			for lcobj_name in r_lcset.get_lcobj_names():
				curve_df = {}
				lcobj = r_lcset[lcobj_name]
				for b in lcobj.bands:
					lcobjb = lcobj.get_b(b)
					for i in range(len(lcobjb)):
						curve_df[f'{b}{i}'] = {
							'day':lcobjb.days[i],
							'obs':lcobjb.obs[i],
							'obse':lcobjb.obse[i],
							'synth':int(lcobjb.synthetic),
						}
				
				curve_df = pd.DataFrame.from_dict(curve_df, orient='index')
				#print(curve_df)
				xy_df.append(curve_df)

			for lcobj_name in s_lcset.get_lcobj_names():
				curve_df = {}
				lcobj = s_lcset[lcobj_name]
				for b in lcobj.bands:
					lcobjb = lcobj.get_b(b)
					for i in range(len(lcobjb)):
						curve_df[f'{b}{i}'] = {
							'day':lcobjb.days[i],
							'obs':lcobjb.obs[i],
							'obse':lcobjb.obse[i],
							'synth':int(lcobjb.synthetic),
						}

				curve_df = pd.DataFrame.from_dict(curve_df, orient='index')
				xy_df.append(curve_df)

			xy_df = pd.concat(xy_df, axis=0)

			### fit
			bar(f'training id: {id} - samples: {len(xy_df)} - features: {len(xy_df.columns)} {list(xy_df.columns)}')
			brf.fit(xy_df[['day', 'obs', 'obse']].values, xy_df[['synth']].values[...,0])

			### evaluate
			eval_lcset = lcdataset[f'{kf}@{test_lcset_name}']

			xy_df = []
			for lcobj_name in eval_lcset.get_lcobj_names():
				curve_df = {}
				lcobj = eval_lcset[lcobj_name]
				for b in lcobj.bands:
					lcobjb = lcobj.get_b(b)
					for i in range(len(lcobjb)):
						curve_df[f'{b}{i}'] = {
							'day':lcobjb.days[i],
							'obs':lcobjb.obs[i],
							'obse':lcobjb.obse[i],
							'synth':int(lcobjb.synthetic),
						}
				
				curve_df = pd.DataFrame.from_dict(curve_df, orient='index')
				#print(curve_df)
				xy_df.append(curve_df)

			xy_df = pd.concat(xy_df, axis=0)

			y_target = xy_df[['synth']].values[...,0]
			y_pred_p = brf.predict_proba(xy_df[['day', 'obs', 'obse']].values)
			y_pred = np.argmax(y_pred_p, axis=-1)
			print(np.mean(y_target==y_pred).astype(np.float))

			### results
			features = ['day', 'obs', 'obse']
			rank = TopRank('features')
			rank.add_list(features, brf.feature_importances_)
			rank.calcule_rank()
			print(rank)
			results_dict[id] = {
				'test_lcset_name':test_lcset_name,
				'class_names':class_names,
				'features':features,
				'rank':rank,
				#'real-recall':recall, # low is good
				#'real-xentropy':xentropy, # high is good
			}

		else:
			raise Exception(f'no train_config: {train_config}')

	bar.done()
	return results_dict, model_ids

def get_mixed_df(indexs, r_x_df, r_y_df, s_x_df, s_y_df,
	replace_class=False,
	):
	r_x_ddf = dd.from_pandas(r_x_df, npartitions=10)
	r_y_ddf = dd.from_pandas(r_y_df, npartitions=10)
	s_x_ddf = dd.from_pandas(s_x_df, npartitions=10)
	s_y_ddf = dd.from_pandas(s_y_df, npartitions=10)
	x_dfs = []
	y_dfs = []
	s_indexs_ = list(s_x_df.index)
	for ki,index in enumerate(indexs):
		s_indexs = [i for i in s_indexs_ if i.split('.')[0]==index]

		r_x_df_i = r_x_ddf.loc[[index]].compute()
		s_x_df_i = s_x_ddf.loc[s_indexs].compute()
		x_dfs += [r_x_df_i]*len(s_x_df_i)+[s_x_df_i]

		r_y_df_i = r_y_ddf.loc[[index]].compute()
		s_y_df_i = s_y_ddf.loc[s_indexs].compute()
		if replace_class:
			r_y_df_i['__y__'].values[:] = 1
			s_y_df_i['__y__'].values[:] = 0
		y_dfs += [r_y_df_i]*len(s_y_df_i)+[s_y_df_i]
		
		if ki>2000:
			break
	x_df = pd.concat(x_dfs, axis=0)
	y_df = pd.concat(y_dfs, axis=0)
	return x_df, y_df

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
			recall = np.sum(y_pred)/len(y_pred)
			xentropy = np.mean(-np.log(y_pred_p[:,1]+1e-10))
			results_dict[id].update({
				'real-recall':recall, # low is good
				'real-xentropy':xentropy, # high is good
			})
			print(recall, xentropy)

	return results_dict