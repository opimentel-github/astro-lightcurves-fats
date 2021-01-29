from __future__ import print_function
from __future__ import division
from . import C_

import turbofats
from .sn_parametric_model_computer import SNModelScipy, get_features_keys
from turbofats import FeatureFunctionLib
import numpy as np
import pandas as pd
from flamingchoripan.progress_bars import ProgressBar
from flamingchoripan.lists import get_list_chunks
from joblib import Parallel, delayed
from .mhps_extractor import MHPSExtractor

###################################################################################################################################################

def get_null_df(features_names):
	return pd.DataFrame.from_dict({'':{f:np.nan for f in features_names}}, orient='index')

def get_spm_features(lcobjb,
	feature_names=None,
	):
	sne_model = SNModelScipy()
	try:
		spm_names = get_features_keys()
		fit_error = sne_model.fit(lcobjb.days, lcobjb.obs, lcobjb.obse)
		#print(fit_error)
		spm_params = sne_model.get_model_parameters()+[fit_error]
		spm_params = {spm:spm_params[k] for k,spm in enumerate(spm_names)}
		#print(spm_params)
		features_df = pd.DataFrame.from_dict({'':spm_params}, orient='index')
	except:
		features_df = get_null_df(spm_names)

	if not feature_names is None:
		features_df = features_df[feature_names]
	return features_df

def get_mhps_features(lcobjb,
	feature_names=None,
	):
	mhps_ex = MHPSExtractor()
	try:
		mag = lcobjb.obs
		magerr = lcobjb.obse
		time = lcobjb.days
		features_df = mhps_ex.compute_feature_in_one_band_(mag, magerr, time)
	except:
		features_df = get_null_df(mhps_ex.get_features_keys())

	if not feature_names is None:
		features_df = features_df[feature_names]
	return features_df

def get_fats_features(lcobjb):
	#if len(lcobjb)==0
	#feature_names = C_.OLD_FEATURES+C_.ALERCE_FEATURES
	feature_names = C_.ALERCE_SPM_FEATURES
	feature_space = turbofats.FeatureSpace(feature_names)
	detections_data = np.concatenate([lcobjb.days[...,None], lcobjb.obs[...,None], lcobjb.obse[...,None]], axis=-1)
	detections = pd.DataFrame(
		data=detections_data,
		columns=['mjd', 'magpsf_corr', 'sigmapsf_corr'],
		index=['']*len(detections_data)
	)
	return feature_space.calculate_features(detections)

###################################################################################################################################################

def get_all_fat_features(lcdataset, lcset_name,
	n_jobs=C_.N_JOBS,
	chunk_size=C_.CHUNK_SIZE,
	backend='threading',
	):
	lcset = lcdataset[lcset_name]
	band_names = lcset.band_names
	features_df = {}
	labels_df = {}
	chunks = get_list_chunks(lcset.get_lcobj_names(), chunk_size)
	bar = ProgressBar(len(chunks))
	for kc,chunk in enumerate(chunks):
		bar(f'lcset_name: {lcset_name} - chunck: {kc} - chunk_size: {chunk_size}')
		results = Parallel(n_jobs=n_jobs, backend=backend)([delayed(get_features)(lcset[lcobj_name], band_names) for lcobj_name in chunk]) # None threading 
		for result, lcobj_name in zip(results, chunk):
			features_df[lcobj_name] = result
			labels_df[lcobj_name] = {
				'__y__':lcset[lcobj_name].y,
				'__fullsynth__':lcset[lcobj_name].all_synthetic(),
				}

	bar.done()
	x = pd.DataFrame.from_dict(features_df, orient='index')
	y = pd.DataFrame.from_dict(labels_df, orient='index')
	return x, y

def get_features(lcobj, band_names):
	df_bdict = {}
	for b in band_names:
		df_to_cat = []
		lcobjb = lcobj.get_b(b)
		
		### fats
		fats_df_b = get_fats_features(lcobjb)
		df_to_cat.append(fats_df_b)

		### spm
		spm_df_b = get_spm_features(lcobjb) # ['SPM_beta', 'SPM_t0', 'SPM_tau_rise', 'SPM_tau_fall']
		df_to_cat.append(spm_df_b)

		### mhps
		mhps_df_b = get_mhps_features(lcobjb) # ['MHPS_low', 'MHPS_ratio']
		df_to_cat.append(mhps_df_b)

		### cat
		df_bdict[b] = pd.concat(df_to_cat, axis=1, sort=True)
		#print(df_bdict[b]) # use this to check first

	for b in band_names:
		df_bdict[b].columns = [f'{c}_{b}' for c in df_bdict[b].columns]

	features_df = pd.concat([df_bdict[b] for b in band_names], axis=1, sort=True)
	features_df = features_df.clip(-abs(C_.NAN_VALUE), abs(C_.NAN_VALUE)).fillna(C_.NAN_VALUE)
	return features_df.to_dict(orient='index')['']