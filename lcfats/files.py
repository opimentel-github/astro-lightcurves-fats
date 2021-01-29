from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir
import os

###################################################################################################################################################

def load_features(filedir):
	df_xy = pd.read_parquet(os.path.abspath(f'{filedir}')) # parquet
	columns = list(df_xy.columns)
	y_columns = ['__y__', '__fullsynth__']
	df_y = df_xy[y_columns]
	df_x = df_xy[[c for c in columns if not c in y_columns]]
	#print(df_y)
	#print(df_x)
	return df_x, df_y

def save_features(df_x, df_y, save_filedir):
	save_rootdir = '/'.join([s for s in save_filedir.split('/')[:-1]])
	create_dir(save_rootdir)
	df_xy =  pd.concat([df_y, df_x], axis=1)
	df_xy.to_parquet(os.path.abspath(f'{save_filedir}')) # parquet