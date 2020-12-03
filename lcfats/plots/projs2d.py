from __future__ import print_function
from __future__ import division
from . import C_

import numpy as np
import matplotlib.pyplot as plt
import flamingchoripan.cuteplots.colors as cc

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
		name_r,_ = obj_r.split('.')
		name_s,id_s = obj_s.split('.')
		if name_r==name_s and not id_s=='0':
			objs.append(obj_s)
			counter += 1
			if counter>=max_synth_samples:
				break
	return objs

###################################################################################################################################################

def plot_projections(maps2d_dict,
	target_class=None,
	figsize:tuple=(10,8),
	alpha=0.75,

	max_samples=3e3,
	max_real_samples:int=250,
	max_synth_samples:int=2,
	):
	fig, axs = plt.subplots(1, 1, figsize=figsize)
	class_names = maps2d_dict['class_names']
	for kc,c in enumerate(class_names):
		ax = axs
		if target_class is None:
			plot_projections_c(ax, maps2d_dict, c, max_samples=max_samples, alpha=alpha)
		else:
			if c==target_class:
				plot_net_projections_c(ax, maps2d_dict, c, alpha=alpha)
			else:
				plot_projections_c(ax, maps2d_dict, c, 'k', max_samples=max_samples, alpha=0.5)

	method_name = maps2d_dict['method_name']
	title = f'{method_name} projection of FATS features'
	#title += f'survey: {lcset_train.survey} - set: {set_name_train}'
	ax.legend()
	ax.set_title(title)
	ax.grid(alpha=0.25)

	fig.tight_layout()
	return fig

def get_standar_style(color, alpha):
	return {
		'facecolors':[color],
		#'edgecolors':'k',
		's':10,
		'alpha':alpha,
		'marker':'o',
		'lw':1.5,
		'linewidth':0.0,
	}

def plot_projections_c(ax, maps2d_dict, c,
	color=None,
	alpha=0.9,
	max_samples=1e3,
	):
	lcobj_names = maps2d_dict['lcobj_names']
	class_names = maps2d_dict['class_names']
	map_x = maps2d_dict['map_x']
	labels = maps2d_dict['y']
	counter = 0

	### plot all
	for idx,lcobj_name in enumerate(lcobj_names):
		if any([not class_names[labels[idx]]==c, counter>max_samples]):
			continue
		color = cc.get_default_colorlist()[labels[idx]] if color is None else color
		map_x_ = map_x[idx]
		ax.scatter(map_x_[0], map_x_[1], **get_standar_style(color, alpha), label=f'{c} (real & synth)' if counter==0 else None)
		counter += 1

def plot_net_projections_c(ax, maps2d_dict, c,
	alpha=0.9,
	max_real_samples:int=200,
	max_synth_samples:int=1,
	):
	lcobj_names = maps2d_dict['lcobj_names']
	class_names = maps2d_dict['class_names']
	map_x = maps2d_dict['map_x']
	labels = maps2d_dict['y']
	real_counter = 0

	### plot all
	for idx,lcobj_name in enumerate(lcobj_names):
		is_synthetic = not lcobj_name.split('.')[-1]=='0'
		if any([not class_names[labels[idx]]==c, real_counter>max_real_samples, is_synthetic]):
			continue

		color = cc.get_default_colorlist()[labels[idx]]
		style = { # circle
			'facecolors':'None',
			'edgecolors':[color],
			#edgecolors='k',
			's':55,
			'alpha':1.,
			'marker':'o',
			'linewidth':1,
			#label=f'{c}' if not has_label[b][c] else None,
		}
		map_x_real = map_x[idx]
		ax.scatter(map_x_real[0], map_x_real[1], **style, label=f'{c} (real)' if real_counter==0 else None)

		### plot synthetics
		lcobj_names_synth = get_synth_objs(lcobj_name, maps2d_dict['lcobj_names_synth'], max_synth_samples)
		for ks,lcobj_name_synth in enumerate(lcobj_names_synth):
			idx = lcobj_names.index(lcobj_name_synth)
			map_x_synth = map_x[idx]
			ax.scatter(map_x_synth[0], map_x_synth[1], **get_standar_style(color, alpha), label=f'{c} (synth)' if real_counter==0 and ks==0 else None)
			line = ax.plot([map_x_real[0], map_x_synth[0]], [map_x_real[1], map_x_synth[1]], alpha=alpha, lw=0.5, c=color, label=f'{c} real-synth bound' if real_counter==0 and ks==0  else None)
			#add_arrow(line[0], color=color)

		real_counter += 1