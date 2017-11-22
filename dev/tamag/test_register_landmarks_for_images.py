#!/usr/bin/env python

import os, sys, commands

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

from msct_register_landmarks import getRigidTransformFromImages
from msct_register_regularized import generate_warping_field

os.chdir('/Users/tamag/data/work_on_registration')

# generate_warping_field('pad_carre.nii.gz', [-162], [185], theta_rot=[0.389])

rotation_matrix, translation_array = getRigidTransformFromImages('pad_carre.nii.gz', 'pad_carre_after_rotation.nii.gz', constraints='xy', metric='MeanSquares', center_rotation='BarycenterImage')
print '\nThe cosine and the sine are: ' + str(rotation_matrix[0][0]) +', '+ str(rotation_matrix[1][0])