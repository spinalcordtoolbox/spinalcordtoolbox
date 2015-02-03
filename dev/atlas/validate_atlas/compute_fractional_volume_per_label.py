__author__ = 'slevy_local'

import sys
path_sct = '/Users/slevy_local/spinalcordtoolbox' #'C:/cygwin64/home/Simon_2/spinalcordtoolbox'
# Append path that contains scripts, to be able to load modules
sys.path.append(path_sct + '/scripts')
import sct_extract_metric

atlas_folder = '/Users/slevy_local/spinalcordtoolbox/data/atlas/'

global param
param = sct_extract_metric.Param()


[label_id, label_name, label_file] = sct_extract_metric.read_info_label(atlas_folder)


