from __future__ import print_function
from __future__ import division
from . import C_

import turbofats
from .sn_parametric_model_computer import SNModelScipy, get_features_keys
from turbofats import FeatureFunctionLib
import numpy as np
import pandas as pd
from flamingchoripan.strings import get_list_chunks
from flamingchoripan.progress_bars import ProgressBar
from joblib import Parallel, delayed

###################################################################################################################################################

def get_all_fat_features(lcdataset, lcset_name,
	chunk_size=20,
	n_jobs=C_.N_JOBS,
	):
	lcset = lcdataset[lcset_name]
	band_names = lcset.band_names
	features_df = {}
	labels_df = {}
	chunks = get_list_chunks(lcset.get_lcobj_names(), chunk_size)
	bar = ProgressBar(len(chunks))
	for kc,chunk in enumerate(chunks):
		bar(f'lcset_name: {lcset_name} - chunck: {kc} - objs: {len(chunk)}')
		results = Parallel(n_jobs=n_jobs)([delayed(get_fat_features)(lcset[lcobj_name], band_names) for lcobj_name in chunk])
		for result, lcobj_name in zip(results, chunk):
			features_df[lcobj_name] = result
			labels_df[lcobj_name] = {'c':lcset[lcobj_name].y}

	bar.done()
	x = pd.DataFrame.from_dict(features_df, orient='index')
	y = pd.DataFrame.from_dict(labels_df, orient='index')
	return x, y

def get_spm_features(lcobjb):
	sne_model = SNModelScipy()
	spm_names = get_features_keys()
	try:
		fit_error = sne_model.fit(lcobjb.days, lcobjb.obs, lcobjb.obse)
		spm_params = sne_model.get_model_parameters()+[fit_error]
	except ValueError:
		spm_params = [np.nan]*len(spm_names)

	spm_params = {spm:spm_params[k] for k,spm in enumerate(spm_names)}
	return pd.DataFrame.from_dict({'':spm_params}, orient='index')

def get_fat_features(lcobj, band_names):
	df_bdict = {}
	for b in band_names:
		lcobjb = lcobj.get_b(b)
		feature_space = turbofats.FeatureSpace(C_.OLD_FEATURES+C_.ALERCE_FEATURES)

		### fats
		detections_data = np.concatenate([lcobjb.days[...,None], lcobjb.obs[...,None], lcobjb.obse[...,None]], axis=-1)
		detections = pd.DataFrame(
			data=detections_data,
			columns=['mjd', 'magpsf_corr', 'sigmapsf_corr'],
			index=['']*len(detections_data)
		)
		features_df_b = feature_space.calculate_features(detections)

		### SPM
		spm_df_b = get_spm_features(lcobjb)
		df_bdict[b] = pd.concat([features_df_b, spm_df_b], axis=1, sort=True)
	
	for b in band_names:
		df_bdict[b].columns = [f'{c}_{b}' for c in df_bdict[b].columns]

	features_df = pd.concat([df_bdict[b] for b in band_names], axis=1, sort=True)
	features_df = features_df.fillna(C_.NAN_VALUE)
	return features_df.to_dict(orient='index')['']