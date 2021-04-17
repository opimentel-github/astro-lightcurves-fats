#!/usr/bin/env python3
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../flaming-choripan') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module
sys.path.append('../../astro-lightcurves-fats') # or just install the module

if __name__== '__main__':
	### parser arguments
	import argparse
	from flamingchoripan.prints import print_big_bar

	parser = argparse.ArgumentParser('usage description')
	parser.add_argument('-method',  type=str, default='.', help='method')
	parser.add_argument('-mids',  type=str, default='0-50', help='initial_id-final_id')
	parser.add_argument('-kf',  type=str, default='.', help='kf')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from lchandler import C_

	rootdir = '../../surveys-save'
	filedirs = search_for_filedirs(rootdir, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lcfats.files import load_features
	from flamingchoripan.progress_bars import ProgressBar
	from lcfats.classifiers import train_classifier, evaluate_classifier
	import pandas as pd

	kfs = [str(kf) for kf in range(0,3)] if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods

	for kf in kfs:
		for method in methods:
			filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={method}.splcds'
			filedict = get_dict_from_filedir(filedir)
			rootdir = filedict['_rootdir']
			cfilename = filedict['_cfilename']
			lcdataset = load_pickle(filedir)
			lcset_info = lcdataset['raw'].get_info()
			print(lcdataset)

			for train_config in ['r', 's', 'r+s']:
				###################################################################################################################################################
				### IDS
				model_ids = list(range(*[int(mi) for mi in main_args.mids.split('-')]))
				bar = ProgressBar(len(model_ids))
				for ki,model_id in enumerate(model_ids): # IDS
					if train_config=='r':
						train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{kf}@train.df')

					if train_config=='s':
						train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{kf}@train.{method}.df')

					if train_config=='r+s':
						train_df_x_r, train_df_y_r = load_features(f'../save/fats/{cfilename}/{kf}@train.df')
						train_df_x_s, train_df_y_s = load_features(f'../save/fats/{cfilename}/{kf}@train.{method}.df')
						repeats = len(train_df_x_s)//len(train_df_x_r)
						train_df_x = pd.concat([train_df_x_r]*repeats+[train_df_x_s], axis=0)
						train_df_y = pd.concat([train_df_y_r]*repeats+[train_df_y_s], axis=0)

					val_df_x, val_df_y = load_features(f'../save/fats/{cfilename}/{kf}@val.df')
					test_df_x, test_df_y = load_features(f'../save/fats/{cfilename}/{kf}@test.df')

					bar(f'kf={kf} - method={method} - train_config={train_config} - model_id={model_id} - samples={len(train_df_y)} - features={len(train_df_x.columns)}')
					#print(list(train_df_x.columns))
					fit_kwargs = {}
					brf = train_classifier(train_df_x, train_df_y, **fit_kwargs)

					results_val = evaluate_classifier(brf, val_df_x, val_df_y, lcset_info, **fit_kwargs)
					save_pickle(f'../save/exp=rf_eval~train_config={train_config}/{cfilename}/{kf}@val/id={model_id}.df', results_val)

					results_test = evaluate_classifier(brf, test_df_x, test_df_y, lcset_info, **fit_kwargs)
					save_pickle(f'../save/exp=rf_eval~train_config={train_config}/{cfilename}/{kf}@test/id={model_id}.df', results_test)

				bar.done()


