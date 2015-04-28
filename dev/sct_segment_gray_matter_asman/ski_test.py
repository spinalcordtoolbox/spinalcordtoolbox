#!/usr/bin/env python

from skimage import transform as tf
from skimage import data
from msct_image import Image
from math import pi


im0 = Image('data_by_slice/slice_0_im.nii.gz')
seg0 = Image('data_by_slice/slice_0_seg.nii.gz')
'''
transfo = tf.SimilarityTransform(scale=1, rotation=pi / 2, translation=(0, 1))
print transfo.params

im0.changeType('uint8')
print im0.data

text = data.text()
im_moved = tf.warp(transfo, text)  # im0.data)

print im_moved

Image(param=im_moved, absolutepath='moved_slice_0.nii.gz').save()
'''

text = data.text()

seg0.changeType('uint8')
test = seg0.data

Image(param=test, absolutepath='converted_slice_0_seg.nii.gz').save()

im0.changeType('uint32')
test_im = im0.data

Image(param=test_im, absolutepath='converted_slice_0.nii.gz').save()


tform = tf.AffineTransform(scale=[1.5, 2], rotation=pi / 4, translation=(test.shape[0] / 2, -1.2))
tform2 = tf.AffineTransform(scale=[1, 1], rotation=0, translation=(0, 0))

seg_rotated = tf.warp(test, tform2)
seg_back_rotated = tf.warp(seg_rotated, tform2.inverse)


Image(param=seg_rotated, absolutepath='affine_moved_slice_0_seg.nii.gz').save()
seg_back = Image(param=seg_back_rotated, absolutepath='affine_back_moved_slice_0_seg.nii.gz').save()

im_rotated = tf.warp(test_im, tform)
im_back_rotated = tf.warp(im_rotated, tform.inverse)


Image(param=im_rotated, absolutepath='affine_moved_slice_0.nii.gz').save()
Image(param=im_back_rotated, absolutepath='affine_back_moved_slice_0.nii.gz').save()
