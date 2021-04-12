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
	parser.add_argument('-kf',  type=str, default='0', help='kf')
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

	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods

	for method in methods:
		filedir = f'../../surveys-save/survey=alerceZTFv7.1°bands=gr°mode=onlySNe°method={method}.splcds'
		filedict = get_dict_from_filedir(filedir)
		rootdir = filedict['__rootdir']
		cfilename = filedict['__cfilename']
		lcdataset = load_pickle(filedir)
		lcset_info = lcdataset['raw'].get_info()
		print(lcdataset)

		#for train_config in ['r', 's', 'r+s']:
		for train_config in ['r', 's']:
			###################################################################################################################################################
			### IDS
			model_ids = list(range(*[int(k) for k in main_args.mids.split('-')]))
			bar = ProgressBar(len(model_ids))
			for ki,model_id in enumerate(model_ids): # IDS
				if train_config=='r':
					train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.df')

				if train_config=='s':
					train_df_x, train_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.{method}.df')

				if train_config=='r+s':
					pass

				val_df_x, val_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@val.df')
				test_df_x, test_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@test.df')

				bar(f'method={method} - train_config={train_config} - model_id={model_id} - samples={len(train_df_y)} - features={len(train_df_x.columns)}')
				#print(list(train_df_x.columns))
				fit_kwargs = {}
				brf = train_classifier(train_df_x, train_df_y, **fit_kwargs)

				results_val = evaluate_classifier(brf, val_df_x, val_df_y, lcset_info, **fit_kwargs)
				save_pickle(f'../save/exp=rf_eval°train_config={train_config}/{cfilename}/{main_args.kf}@val/id={model_id}.df', results_val)

				results_test = evaluate_classifier(brf, test_df_x, test_df_y, lcset_info, **fit_kwargs)
				save_pickle(f'../save/exp=rf_eval°train_config={train_config}/{cfilename}/{main_args.kf}@test/id={model_id}.df', results_test)

			bar.done()



