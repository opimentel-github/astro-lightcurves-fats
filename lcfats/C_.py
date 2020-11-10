import sys
import numpy as np

###################################################################################################################################################

BAR_FULL_MODE = '%(current)d/%(total)d %(bar)s %(percent)3d%% - %(remaining)d remaining - %(string)s'
DEFAULT_PRINT_OUTPUT = sys.stdout # sys.stdout, sys.stderr

FILEDIR = '*filedir*' # ../../save/data.txt
ROOTDIR = '*rootdir*' # ../../save/
FILENAME = '*filename*' # data.txt
CFILENAME = '*cfilename*' # data
FEXT = '*fext*' # txt

FILESIZE_FACTOR = 1e-6 # in mbs

KEY_KEY_SEP_CHAR = '_'
KEY_VALUE_SEP_CHAR = '-'

BAR_SIZE = None # for auto width
JUPYTER_NOTEBOOK_BAR_SIZE = 100
BAR_LAST_CHAR = '>'
MIDDLE_LINE_CHAR = '─'
BOT_SQUARE_CHAR ='▄'
TOP_SQUARE_CHAR = '▀'
BAR_CHAR= '-'
DOBLE_BAR_CHAR = '═'
NAN_CHAR = '─'

DELTA_LATEXCHAR = '$\Delta$'

###################################################################################################################################################
### cuteplots

PLOT_STD_ALPHA = 0.25
PLOT_GRID_ALPHA = 0.25
PLOT_MPLIB_FIGSIZE = (6,4)
PLOT_FIGSIZE = (11,4)
PLOT_FIGSIZE_BOX = (6,4)
PLOT_FIGSIZE_CMAP = (10,8)
PLOT_DPI = 80

AN_VIDEO_QUALITY = 10
AN_LOAD_IMAGE_FEXT = 'jpg'
AN_SAVE_IMAGE_FEXT = 'png'
AN_VIDEO_FEXT = 'mp4'
AN_SEGS_OFFSET = 0.75

DEFAULT_CMAP = 'cc_favs2'
SAMPLES_TEXT = 'N'

###################################################################################################################################################
### datascience

CM_FORMAT = np.uint32
SERROR_LATEXCHAR = '$SE$'
STD_LATEXCHAR = '$\sigma$'

###################################################################################################################################################
### latex
DEFAULT_MODEL_COL_NAME = '*model*'

KEY_KEY_SEP_CHAR = '_'
KEY_VALUE_SEP_CHAR = '-'

#NEW_LATEXCHAR = '\\_'
NEW_LATEXCHAR = '_'
SEP_LATEXCHAR = '&'
SUM_MINUS_LATEXCHAR = '$\\pm$'
NAN_LATEXCHAR = '--'
LATEX_HLINE = '\\hline'

C_HARD_GRAY = '#343434'
C_LIGHT_GRAY = '#ebebeb'

MEAN_DECIMALS = 1
STD_DECIMALS = 2
