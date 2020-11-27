from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir

###################################################################################################################################################

def load_features(load_filedir):
	pass

def save_features(df_x, df_y, save_filedir):
	create_dir('/'.join([s for s in save_filedir.split('/')[:-1]]))
	df_x.to_parquet(f'{save_filedir}.x') # parquet
	df_y.to_parquet(f'{save_filedir}.y') # parquet