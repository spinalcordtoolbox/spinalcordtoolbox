#!/usr/bin/env python


import sys, os

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))


from msct_register_regularized import generate_warping_field
import sct_utils as sct


path = '/Users/tamag/Desktop/sliceregaffine/negativescaling/tmp.150717110850/tmp.150717110853'
fname_dest = 'src_reg_z0000.nii'


os.chdir(path)

print '\nGet image dimensions of destination image...'
nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(fname_dest)

x_trans = [0 for i in range(nz)]
y_trans= [0 for i in range(nz)]

generate_warping_field(fname_dest, x_trans=x_trans, y_trans=y_trans, fname='warp_null.nii.gz')
