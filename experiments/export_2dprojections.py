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
	parser.add_argument('-mode',  type=str, default='all', help='mode')
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
	from lcfats.map2d import get_fitted_maps2d
	import matplotlib.pyplot as plt
	from lcfats.plots.projs2d import plot_projections
	from flamingchoripan.progress_bars import ProgressBar
	from flamingchoripan.cuteplots.utils import save_fig
	from flamingchoripan.strings import get_string_from_dict
	from flamingchoripan.datascience.grid_search import GDIter, GridSeacher

	kfs = list(range(0, 5)) if main_args.kf=='.' else main_args.kf
	kfs = [kfs] if isinstance(kfs, str) else kfs
	methods = ['linear-fstw', 'bspline-fstw', 'spm-mle-fstw', 'spm-mle-estw', 'spm-mcmc-fstw', 'spm-mcmc-estw'] if main_args.method=='.' else main_args.method
	methods = [methods] if isinstance(methods, str) else methods

	for kf in kfs:
		for method in methods:
			grid_params = {
				'min_dist':0.5,
				'n_neighbors':GDIter(*np.linspace(10, 50, 8).astype(np.int)[::-1]),
			}
			gs = GridSeacher(grid_params)
			bar = ProgressBar(len(gs))
			for params in gs:
				bar(f'method={method} - kf={kf} - params={params}')
				lcset_name = f'{kf}@train.{method}'
				load_rootdir = f'../save/fats/{cfilename}~method={method}'
				map_kwargs = {
					'features_mode':main_args.mode,
					'proj_mode':'pca+umap',
					'min_dist':params['min_dist'],
					'n_neighbors':params['n_neighbors'],
				}
				maps2d_dict = get_fitted_maps2d(lcdataset, lcset_name, load_rootdir, **map_kwargs)
				class_names = lcdataset['raw'].class_names
				for c in [None]+class_names:
					fig = plot_projections(maps2d_dict, lcdataset, lcset_name, c)
					if c is None:
						save_filedir = f'../save/exp=umap~mode={main_args.mode}/{cfilename}~method={method}/{lcset_name}/{get_string_from_dict(params)}.png'
					else:
						save_filedir = f'../save/exp=umap~mode={main_args.mode}/{cfilename}~method={method}/{lcset_name}/{get_string_from_dict(params)}/{c}.png'
					save_fig(save_filedir, fig)
					
			bar.done()