from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import fuzzytools.cuteplots.colors as cc
from fuzzytools.datascience.ranks import TopRank

###################################################################################################################################################

def add_arrow(line, position=None, direction='right', size=15, color=None):
	"""
	add an arrow to a line.

	line:       Line2D object
	position:   x-position of the arrow. If None, mean of xdata is taken
	direction:  'left' or 'right'
	size:       size of the arrow in fontsize points
	color:      if None, line color is taken.
	"""
	if color is None:
		color = line.get_color()

	xdata = line.get_xdata()
	ydata = line.get_ydata()

	if position is None:
		position = xdata.mean()
	# find closest index
	#start_ind = np.argmin(np.absolute(xdata - position))
	start_ind = 0
	if direction == 'right':
		end_ind = start_ind + 1
	else:
		end_ind = start_ind - 1

	line.axes.annotate('',
		xytext=(xdata[start_ind], ydata[start_ind]),
		xy=(xdata[end_ind], ydata[end_ind]),
		arrowprops=dict(arrowstyle="->", color=color),
		size=size
	)

def get_synth_objs(obj_r, objs_s,
	max_synth_samples=1,
	):
	#print(obj_r, objs_s[:10])
	objs = []
	counter = 0
	for k,obj_s in enumerate(objs_s):
		name_r = obj_r
		name_s,id_s = obj_s.split('.')
		if name_r==name_s:
			objs.append(obj_s)
			counter += 1
			if counter>=max_synth_samples:
				break
	return objs

def get_standar_style(color, alpha):
	d = {
		'facecolors':[color],
		#'edgecolors':'k',
		's':5,
		'alpha':alpha,
		'marker':'o',
		'lw':1.5,
		'linewidth':0.0,
	}
	return d

###################################################################################################################################################

def plot_projections(maps2d_dict, lcdataset, s_lcset_name,
	target_class=None,
	figsize:tuple=(14,12),
	marker_alpha=0.8,
	r_max_samples=np.inf,
	):
	r_lcset = lcdataset[s_lcset_name.split('.')[0]]
	fig, axs = plt.subplots(1, 1, figsize=figsize)
	class_names = maps2d_dict['class_names']
	rank_names = []
	for kc,c in enumerate(class_names):
		ax = axs
		p_kwargs = {
			'r_max_samples':r_max_samples,
			'marker_alpha':marker_alpha,
		}
		if target_class is None:
			plot_projections_c(ax, maps2d_dict, c, **p_kwargs)
		else:
			if c==target_class:
				rank_names = plot_projections_c(ax, maps2d_dict, c, uses_net=True, **p_kwargs)
			else:
				plot_projections_c(ax, maps2d_dict, c, color='k', **p_kwargs)

	method_name = maps2d_dict['method_name']
	title = ''
	title += f'{method_name}'+'\n'
	title += f'survey={r_lcset.survey}-{"".join(r_lcset.band_names)} [{s_lcset_name}]'+'\n'
	if len(rank_names)>0:
		title += f'real-synth top distance=[{", ".join(rank_names)}]'+'\n'  
	ax.legend(loc='upper right', prop={"size":15})
	ax.set_title(title[:-1])
	ax.grid(alpha=0.25)
	fig.tight_layout()
	return fig

def plot_projections_c(ax, maps2d_dict, c,
	color=None,
	marker_alpha=0.9,
	r_max_samples=np.inf,
	s_max_samples=1,
	uses_net=False,
	net_alpha=0.25,
	rank=3,
	fontsize=14,
	):
	net_alpha = 0.05 if c in ['SNIa'] else net_alpha # fixme
	map_lcobj_names = maps2d_dict['map_lcobj_names']
	class_names = maps2d_dict['class_names']
	map_x = maps2d_dict['map_x']
	labels = maps2d_dict['y']
	r_counter = 0
	dist_rank = TopRank()

	### plot all
	for idx,lcobj_name in enumerate(map_lcobj_names):
		if '.' in lcobj_name: # is synthetic
			continue
		if r_counter>r_max_samples:
			continue
		if not class_names[labels[idx]]==c:
			continue

		color = cc.get_default_colorlist()[labels[idx]] if color is None else color
		r_style = { # circle
			'facecolors':'None',
			'edgecolors':[color],
			#edgecolors='k',
			's':30,
			'alpha':marker_alpha,
			'marker':'o',
			'linewidth':1,
			#label=f'{c}' if not has_label[b][c] else None,
		}
		map_x_real = map_x[idx]
		ax.scatter(map_x_real[0], map_x_real[1], **r_style, label=f'{c} [real]' if r_counter==0 else None)

		### plot synthetics
		lcobj_names_synth = get_synth_objs(lcobj_name, maps2d_dict['s_lcobj_names'], s_max_samples)
		for ks,lcobj_name_synth in enumerate(lcobj_names_synth):
			idx = map_lcobj_names.index(lcobj_name_synth)
			map_x_synth = map_x[idx]
			ax.scatter(map_x_synth[0], map_x_synth[1], **get_standar_style(color, marker_alpha), label=f'{c} [synth]' if r_counter==0 and ks==0 else None)
			if uses_net:
				dx = map_x_real[0]-map_x_synth[0]
				dy = map_x_real[1]-map_x_synth[1]
				dist = dx**2+dy**2
				line = ax.plot([map_x_real[0], map_x_synth[0]], [map_x_real[1], map_x_synth[1]], alpha=net_alpha, lw=0.5, c=color)
				ax.plot([None], [None], alpha=marker_alpha, lw=0.5, c=color, label=f'{c} real-synth' if r_counter==0 and ks==0  else None)
				dist_rank.append(lcobj_name_synth, dist, {'pos':(map_x_synth[0], map_x_synth[1]), 'line':line[0]})

		r_counter += 1

	### rank
	rank_names = []
	if uses_net:
		dist_rank.calcule()
		for k in range(0, rank):
			name, value, info = dist_rank[k]
			rank_names += [name]
			info['line'].set_alpha(marker_alpha)
			txt = ax.text(*info['pos'], name, horizontalalignment='center', fontsize=fontsize, c=color)
			txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

	return rank_names