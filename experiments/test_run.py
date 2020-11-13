#!/usr/bin/env python3
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../flaming-choripan') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-method',  type=str, default=None, help='method')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from lchandler import C_

	root_folder = '../../astro-lightcurves-handler/save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lchandler import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = '../../astro-lightcurves-handler/save/alerceZTFv5.1/survey-alerceZTFv5.1_bands-gr_mode-onlySNe_kfid-0.splcds'
	filedir = '../../astro-lightcurves-handler/save/alerceZTFv7.1/survey-alerceZTFv7.1_bands-gr_mode-onlySNe_kfid-0.splcds'

	filedic = get_dict_from_filedir(filedir)
	root_folder = filedic['*rootdir*']
	cfilename = filedic['*cfilename*']
	lcdataset = load_lcdataset(filedir)
	print(lcdataset['raw'].keys())
	print(lcdataset['raw'].get_random_lcobj(False).keys())
	print(lcdataset)

	###################################################################################################################################################
	import turbofats

	feature_space = turbofats.NewFeatureSpace(feature_list=['PeriodLS_v2', 'Period_fit_v2', 'Harmonics'])

	detections_data = np.stack(
		[
			time.flatten(),
			magnitude.flatten(),
			error
		],
		axis=-1
	)
	detections = pd.DataFrame(
		data=detections_data,
		columns=['mjd', 'magpsf_corr', 'sigmapsf_corr'],
		index=['asdf'] * len(detections_data)
	)
	feature_values = feature_space.calculate_features(detections)
	print(feature_values)