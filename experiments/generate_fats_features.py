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
	parser.add_argument('-method',  type=str, default='all', help='method')
	#parser.add_argument('-set',  type=str, default='train', help='set')
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
	from lchandler import C_

	def load_lcdataset(filename):
		assert filename.split('.')[-1]==C_.EXT_SPLIT_LIGHTCURVE
		return load_pickle(filename)

	filedir = f'../../surveys-save/alerceZTFv7.1/survey-alerceZTFv7.1_bands-gr_mode-onlySNe_method-{main_args.method}.splcds'

	filedict = get_dict_from_filedir(filedir)
	root_folder = filedict['*rootdir*']
	cfilename = filedict['*cfilename*']
	survey = filedict['survey']
	lcdataset = load_lcdataset(filedir)
	print(lcdataset['raw'].keys())
	print(lcdataset['raw'].get_random_lcobj(False).keys())
	print(lcdataset)

	###################################################################################################################################################
	from lcfats.extractors import get_all_fat_features
	from lcfats.files import save_features

	methods = main_args.method
	if methods=='all':
		methods = ['linear', 'bspline', 'uniformprior', 'curvefit', 'mcmc']

	if isinstance(methods, str):
		methods = [methods]

	for method in methods:
		lcset_names = [lcset_name for lcset_name in lcdataset.get_lcset_names() if not 'raw' in lcset_name] # ignore all raws because we are not using these
		for lcset_name in lcset_names:
			df_x, df_y = get_all_fat_features(lcdataset, lcset_name)
			save_rootdir = f'../save/{survey}/{cfilename}'
			save_filedir = f'{save_rootdir}/{lcset_name}.ftres'
			save_features(df_x, df_y, save_filedir)


