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
	parser.add_argument('-mids',  type=str, default='0-10', help='initial_id-final_id')
	parser.add_argument('-mode',  type=str, default='all', help='mode')
	parser.add_argument('-kf',  type=str, default='.', help='kf')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle, get_dict_from_filedir

	filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe.splcds'
	filedict = get_dict_from_filedir(filedir)
	rootdir = filedict['_rootdir']
	cfilename = filedict['_cfilename']
	survey = filedict['survey']
	lcdataset = load_pickle(filedir)
	print(lcdataset)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir
	from lcfats.files import load_features
	from flamingchoripan.progress_bars import ProgressBar
	from lcfats.classifiers import train_classifier, evaluate_classifier
	import pandas as pd

	kfs = lcdataset.kfolds if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	#methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
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
			#for train_config in ['r', 's']:
				###################################################################################################################################################
				### IDS
				model_ids = list(range(*[int(mi) for mi in main_args.mids.split('-')]))
				bar = ProgressBar(len(model_ids))
				for ki,model_id in enumerate(model_ids): # IDS
					if train_config=='r':
						train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{kf}@train.df', main_args.mode)

					if train_config=='s':
						train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{kf}@train.{method}.df', main_args.mode)

					if train_config=='r+s':
						train_df_x_r, train_df_y_r = load_features(f'../save/fats/{cfilename}/{kf}@train.df', main_args.mode)
						train_df_x_s, train_df_y_s = load_features(f'../save/fats/{cfilename}/{kf}@train.{method}.df', main_args.mode)
						repeats = len(train_df_x_s)//len(train_df_x_r)
						train_df_x = pd.concat([train_df_x_r]*repeats+[train_df_x_s], axis=0)
						train_df_y = pd.concat([train_df_y_r]*repeats+[train_df_y_s], axis=0)

					val_df_x, val_df_y = load_features(f'../save/fats/{cfilename}/{kf}@val.df', main_args.mode)
					test_df_x, test_df_y = load_features(f'../save/fats/{cfilename}/{kf}@test.df', main_args.mode)

					#print(list(train_df_x.columns))
					fit_kwargs = {}
					features = list(train_df_x.columns)
					brf_d = train_classifier(train_df_x, train_df_y, **fit_kwargs)

					results_val = evaluate_classifier(brf_d, val_df_x, val_df_y, lcset_info, **fit_kwargs)
					save_pickle(f'../save/exp=rf_eval~train_config={train_config}~mode={main_args.mode}/{cfilename}/{kf}@val/id={model_id}.df', results_val)

					results_test = evaluate_classifier(brf_d, test_df_x, test_df_y, lcset_info, **fit_kwargs)
					save_pickle(f'../save/exp=rf_eval~train_config={train_config}~mode={main_args.mode}/{cfilename}/{kf}@test/id={model_id}.df', results_test)

					accu = results_test['metrics_dict']['b-accuracy']
					bar(f'kf={kf} - method={method} - train_config={train_config} - model_id={model_id} -accu={accu} - samples={len(train_df_y)} - features={features}({len(features)}#)')

				bar.done()



