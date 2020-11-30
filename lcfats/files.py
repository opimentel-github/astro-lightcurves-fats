from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir
import os

###################################################################################################################################################

def load_features(load_filedir):
	x_df = pd.read_parquet(os.path.abspath(f'{load_filedir}.x')) # parquet
	y_df = pd.read_parquet(os.path.abspath(f'{load_filedir}.y')) # parquet
	return x_df, y_df

def save_features(df_x, df_y, save_filedir):
	create_dir('/'.join([s for s in save_filedir.split('/')[:-1]]))
	df_x.columns = df_x.columns.astype(str)
	df_y.columns = df_y.columns.astype(str)
	df_x.to_parquet(os.path.abspath(f'{save_filedir}.x')) # parquet
	df_y.to_parquet(os.path.abspath(f'{save_filedir}.y')) # parquet