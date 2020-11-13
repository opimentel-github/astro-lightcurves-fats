from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir

###################################################################################################################################################

def save_features_df(df, lcdataset, lcset_name):
	survey_name = lcdataset[lcset_name].survey
	save_root_dir = f'../save/{survey_name}'
	create_dir(save_root_dir)
	df.to_parquet(f'{save_root_dir}/{lcset_name}.parquet')