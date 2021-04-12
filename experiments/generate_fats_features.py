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
	parser.add_argument('-method',  type=str, default='.', help='method')
	parser.add_argument('-kf',  type=str, default='.', help='kf')
	main_args = parser.parse_args()
	print_big_bar()

	###################################################################################################################################################
	from flamingchoripan.files import search_for_filedirs
	from lchandler import C_

	root_folder = '../../surveys-save'
	filedirs = search_for_filedirs(root_folder, fext=C_.EXT_SPLIT_LIGHTCURVE)

	###################################################################################################################################################
	import numpy as np
	from flamingchoripan.files import load_pickle, save_pickle
	from flamingchoripan.files import get_dict_from_filedir

	kfs = [str(kf) for kf in range(0,3)] if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods

	for method in methods:
		filedir = f'../../surveys-save/alerceZTFv7.1/survey=alerceZTFv7.1°bands=gr°mode=onlySNe°method={method}.splcds'
		filedict = get_dict_from_filedir(filedir)
		root_folder = filedict['__rootdir']
		cfilename = filedict['__cfilename']
		survey = filedict['survey']
		lcdataset = load_pickle(filedir)
		print(lcdataset)

		###################################################################################################################################################
		from lcfats.extractors import get_all_fat_features
		from lcfats.files import save_features
	
		for lcset_name in lcdataset.get_lcset_names():
			if len(lcdataset[lcset_name])==0:
				continue
			if 'raw' in lcset_name:
				continue
			kf = lcset_name[0] if '@' in lcset_name else None
			if not kf is None:
				if not kf in kfs:
					continue

			df_x, df_y = get_all_fat_features(lcdataset, lcset_name)
			save_rootdir = f'../save/{survey}/{cfilename}'
			save_filedir = f'{save_rootdir}/{lcset_name}.ftres'
			save_features(df_x, df_y, save_filedir)



