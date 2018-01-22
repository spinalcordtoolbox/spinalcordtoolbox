__author__ = 'julien'

import SimpleITK as sitk
import numpy as np

# from __future__ import print_function

import matplotlib.pyplot as plt
# %matplotlib inline
# from IPython.html.widgets import interact, fixed

OUTPUT_DIR = "Output"

print(sitk.Version())

import numpy as np

def point2str(point, precision=1):
    """
    Format a point for printing, based on specified precision with trailing zeros. Uniform printing for vector-like data
    (tuple, numpy array, list).

    Args:
        point (vector-like): nD point with floating point coordinates.
        precision (int): Number of digits after the decimal point.
    Return:
        String represntation of the given point "xx.xxx yy.yyy zz.zzz...".
    """
    return ' '.join(format(c, '.{0}f'.format(precision)) for c in point)


def uniform_random_points(bounds, num_points):
    """
    Generate random (uniform withing bounds) nD point cloud. Dimension is based on the number of pairs in the bounds input.

    Args:
        bounds (list(tuple-like)): list where each tuple defines the coordinate bounds.
        num_points (int): number of points to generate.

    Returns:
        list containing num_points numpy arrays whose coordinates are within the given bounds.
    """
    internal_bounds = [sorted(b) for b in bounds]
         # Generate rows for each of the coordinates according to the given bounds, stack into an array,
         # and split into a list of points.
    mat = np.vstack([np.random.uniform(b[0], b[1], num_points) for b in internal_bounds])
    return list(mat[:len(bounds)].T)


def target_registration_errors(tx, point_list, reference_point_list):
  """
  Distances between points transformed by the given transformation and their
  location in another coordinate system. When the points are only used to evaluate
  registration accuracy (not used in the registration) this is the target registration
  error (TRE).
  """
  return [np.linalg.norm(np.array(tx.TransformPoint(p)) -  np.array(p_ref))
          for p,p_ref in zip(point_list, reference_point_list)]


def print_transformation_differences(tx1, tx2):
    """
    Check whether two transformations are "equivalent" in an arbitrary spatial region
    either 3D or 2D, [x=(-10,10), y=(-100,100), z=(-1000,1000)]. This is just a sanity check,
    as we are just looking at the effect of the transformations on a random set of points in
    the region.
    """
    if tx1.GetDimension()==2 and tx2.GetDimension()==2:
        bounds = [(-10,10),(-100,100)]
    elif tx1.GetDimension()==3 and tx2.GetDimension()==3:
        bounds = [(-10,10),(-100,100), (-1000,1000)]
    else:
        raise ValueError('Transformation dimensions mismatch, or unsupported transformation dimensionality')
    num_points = 10
    point_list = uniform_random_points(bounds, num_points)
    tx1_point_list = [ tx1.TransformPoint(p) for p in point_list]
    differences = target_registration_errors(tx2, point_list, tx1_point_list)
    print(tx1.GetName()+ '-' +
          tx2.GetName()+
          ':\tminDifference: {:.2f} maxDifference: {:.2f}'.format(min(differences), max(differences)))


# # SimpleITK points represented by vector-like data structures.
# point_tuple = (9.0, 10.531, 11.8341)
# point_np_array = np.array([9.0, 10.531, 11.8341])
# point_list = [9.0, 10.531, 11.8341]
#
# print(point_tuple)
# print(point_np_array)
# print(point_list)
#
# # Uniform printing with specified precision.
# precision = 2
# print(point2str(point_tuple, precision))
# print(point2str(point_np_array, precision))
# print(point2str(point_list, precision))
#
#
# # ========================================
# # TranslationTransform
# # ========================================
#
# # A 3D translation. Note that you need to specify the dimensionality, as the sitk TranslationTransform
# # represents both 2D and 3D translations.
# dimension = 3
# offset =(1,2,3) # offset can be any vector-like data
# translation = sitk.TranslationTransform(dimension, offset)
# print(translation)
#
# # Transform a point and use the inverse transformation to get the original back.
# point = [10, 11, 12]
# transformed_point = translation.TransformPoint(point)
# translation_inverse = translation.GetInverse()
# print('original point: ' + point2str(point) + '\n'
#       'transformed point: ' + point2str(transformed_point) + '\n'
#       'back to original: ' + point2str(translation_inverse.TransformPoint(transformed_point)))
#
#
# # ========================================
# # Euler2DTransform
# # ========================================
#
# point = [10, 11]
# rotation2D = sitk.Euler2DTransform()
# rotation2D.SetTranslation((7.2, 8.4))
# rotation2D.SetAngle(np.pi/2)
# print('original point: ' + point2str(point) + '\n'
#       'transformed point: ' + point2str(rotation2D.TransformPoint(point)))
#
# # Change the center of rotation so that it coincides with the point we want to
# # transform, why is this a unique configuration?
# rotation2D.SetCenter(point)
# print('original point: ' + point2str(point) + '\n'
#       'transformed point: ' + point2str(rotation2D.TransformPoint(point)))

# Working with images

# register with ANTS2d

# run affine/rigid
getstatusoutput('isct_antsRegistration -d 2 -t Rigid[0.5] -m MeanSquares[phantom_dest.nii.gz,phantom_src.nii.gz,1,4] -c 50 -f 1 -s 5mm -o [warp2d,phantom_src_reg.nii.gz] -n NearestNeighbor -v')
# run syn with no iteration (just to get a warping field)
getstatusoutput('isct_antsRegistration -d 2 -t SyN[1, 1, 1] -c 0 -m MI[phantom_dest.nii.gz,phantom_src.nii.gz,1,32] -o warp2d -f 1 -s 0')
# concatenates mat transfo and warping field
getstatusoutput('isct_ComposeMultiTransform 2 warp2d_dest2src.nii.gz -R phantom_dest.nii.gz warp2d0Warp.nii.gz warp2d0GenericAffine.mat')
# now apply warping field for sanity check
getstatusoutput('isct_antsApplyTransforms -d 2 -i phantom_src.nii.gz -r phantom_dest.nii.gz -t warp2d_dest2src.nii.gz -o phantom_src_reg2.nii.gz -n NearestNeighbor')

import sys
# append path that contains scripts, to be able to load modules
sys.path.append('/Users/julien/code/spinalcordtoolbox/scripts')
from msct_image import Image
# open images
im_src = Image('data/src_Z0000.nii').data
im_dest = Image('data/dest_Z0000.nii').data
# display image
from matplotlib.pylab import *
matshow(im_src, fignum=1, cmap=cm.gray), show()
matshow(im_dest, fignum=2, cmap=cm.gray), show()
# open transformation
from scipy.io import loadmat
from math import asin
matfile = loadmat('data/warp2d_00000GenericAffine.mat', struct_as_record=True)
array_transfo = matfile['AffineTransform_double_2_2']
x_displacement = array_transfo[4][0]  # Tx in ITK'S coordinate system
y_displacement = array_transfo[5][0]  # Ty  in ITK'S and fslview's coordinate systems
theta_rotation = asin(array_transfo[2])  # angle of rotation theta in ITK'S coordinate system (minus theta for fslview)

