#!/usr/bin/env python3
# -*- coding: utf-8 -*
import sys
sys.path.append('../') # or just install the module
sys.path.append('../../fuzzy-tools') # or just install the module
sys.path.append('../../astro-lightcurves-handler') # or just install the module
sys.path.append('../../astro-lightcurves-fats') # or just install the module

###################################################################################################################################################
import argparse
from fuzzytools.prints import print_big_bar

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--method',  type=str, default='.')
parser.add_argument('--mid',  type=str, default='.')
parser.add_argument('--kf',  type=str, default='.')
parser.add_argument('--mode',  type=str, default='all')
main_args = parser.parse_args()
print_big_bar()

###################################################################################################################################################
import numpy as np
from fuzzytools.files import load_pickle, save_pickle, get_dict_from_filedir
from lcfats.files import load_features
from fuzzytools.progress_bars import ProgressBar
from lcfats.classifiers import train_classifier, evaluate_classifier
import pandas as pd

filedir = f'../../surveys-save/survey=alerceZTFv7.1~bands=gr~mode=onlySNe~method={main_args.method}.splcds'
filedict = get_dict_from_filedir(filedir)
rootdir = filedict['_rootdir']
cfilename = filedict['_cfilename']
lcdataset = load_pickle(filedir)
lcset_info = lcdataset['raw'].get_info()
lcdataset.only_keep_kf(main_args.kf) # saves ram
print(lcdataset)

for train_config in ['r', 's', 'r+s']:
#for train_config in ['r', 's']:
	print(f'training brf for train_config={train_config}; kf={main_args.kf}; mode={main_args.mode}; method={main_args.method}; mid={main_args.mid}')
	train_df_x_r, train_df_y_r = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.df', main_args.mode)
	train_df_x_s, train_df_y_s = load_features(f'../save/fats/{cfilename}/{main_args.kf}@train.{main_args.method}.df', main_args.mode)
	s_repeats = len(train_df_x_s)//len(train_df_x_r)
	if train_config=='r':
		train_df_x = pd.concat([train_df_x_r]*s_repeats*2, axis='rows')
		train_df_y = pd.concat([train_df_y_r]*s_repeats*2, axis='rows')

	if train_config=='s':
		train_df_x = pd.concat([train_df_x_s]*2, axis='rows')
		train_df_y = pd.concat([train_df_y_s]*2, axis='rows')

	if train_config=='r+s':
		train_df_x = pd.concat([train_df_x_r]*s_repeats+[train_df_x_s], axis='rows')
		train_df_y = pd.concat([train_df_y_r]*s_repeats+[train_df_y_s], axis='rows')

	fit_kwargs = {}
	features = list(train_df_x.columns)
	val_df_x, val_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@val.df', main_args.mode)
	brf_d = train_classifier(train_df_x, train_df_y, val_df_x, val_df_y, lcset_info, **fit_kwargs)

	test_df_x, test_df_y = load_features(f'../save/fats/{cfilename}/{main_args.kf}@test.df', main_args.mode)
	results_test = evaluate_classifier(brf_d, test_df_x, test_df_y, lcset_info, **fit_kwargs)
	save_pickle(f'../save/exp=rf_eval~train_config={train_config}~mode={main_args.mode}/{cfilename}/{main_args.kf}@test/id={main_args.mid}.d', results_test)

	metrics_dict = results_test['metrics_dict']
	brecall = metrics_dict['b-recall']
	print(f'features={features}({len(features)}#)')
	print(f'samples={len(train_df_y)}; b-recall={brecall}')



