from __future__ import print_function
from __future__ import division
from . import C_

import turbofats
from .sn_parametric_model_computer import SNModelScipy, get_features_keys
from turbofats import FeatureFunctionLib
import numpy as np
import pandas as pd
from flamingchoripan.progress_bars import ProgressBar

###################################################################################################################################################

def get_all_fat_features(lcdataset, lcset_name):
	lcset = lcdataset[lcset_name]
	band_names = lcset.band_names
	lcobj_names = lcset.get_lcobj_names()
	features_df = {}
	bar = ProgressBar(len(lcobj_names))
	for k,lcobj_name in enumerate(lcobj_names):
		bar(lcobj_name)
		lcobj = lcset[lcobj_name]
		fats_df = get_fat_features(lcobj, band_names)
		features_df[lcobj_name] = fats_df
		if k>=5:
			#break
			pass
	bar.done()
	return pd.DataFrame.from_dict(features_df, orient='index')

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