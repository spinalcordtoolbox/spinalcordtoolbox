__author__ = 'slevy_local'

import sys

import numpy
import nibabel

path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

import sct_extract_metric

def compute_fract_vol_per_lab(atlas_folder, file_label):

    atlas_folder = '/Users/slevy_local/spinalcordtoolbox/dev/atlas/validate_atlas/cropped_atlas/'
    file_label = 'info_label.txt'


    [label_id, label_name, label_file] = sct_extract_metric.read_label_file(atlas_folder, file_label)
    nb_label = len(label_file)

    fract_volume_per_lab = numpy.zeros((nb_label))

    # compute fractional volume for each label
    for i_label in range(0, nb_label):
        fract_volume_per_lab[i_label] = numpy.sum(nibabel.load(os.path.join(atlas_folder, label_file[i_label)]).get_data())


    print 'Labels\'name:'
    print label_name
    print '\nCorresponding fractional volume:'
    print fract_volume_per_lab

    return label_name, fract_volume_per_lab






