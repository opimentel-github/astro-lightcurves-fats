from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir
import os

###################################################################################################################################################

def save_features(df_x, df_y, save_filedir):
	save_rootdir = '/'.join([s for s in save_filedir.split('/')[:-1]])
	create_dir(save_rootdir)
	df_xy =  pd.concat([df_y, df_x], axis=1)
	df_xy.to_parquet(os.path.abspath(f'{save_filedir}')) # parquet