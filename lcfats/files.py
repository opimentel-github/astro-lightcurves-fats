from __future__ import print_function
from __future__ import division
from . import C_

import pandas as pd
from flamingchoripan.files import create_dir

###################################################################################################################################################

def save_features_df(df, lcdataset, lcset_name, extra_name, save_rootdir):
	survey_name = lcdataset[lcset_name].survey
	new_save_rootdir = f'{save_rootdir}/{survey_name}'
	create_dir(new_save_rootdir)
	df.to_parquet(f'{new_save_rootdir}/{lcset_name}.{extra_name}.parquet')