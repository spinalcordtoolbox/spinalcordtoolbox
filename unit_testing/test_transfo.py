#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for transform stuff

from __future__ import print_function, absolute_import, division

import sys, io, os, time, itertools

import pytest

import numpy as np
import nibabel
import nibabel.orientations

from spinalcordtoolbox.utils import __sct_dir__
sys.path.append(os.path.join(__sct_dir__, 'scripts'))
import sct_utils as sct
import spinalcordtoolbox.image as msct_image
import sct_image
import sct_apply_transfo


def fake_image_custom(data):
    """
    :return: a Nifti1Image (3D) in RAS+ space
    """
    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


def fake_image_sct_custom(data):
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_image_custom(data)
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img


def fake_3dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 10
    - shape[PA] = 20
    - shape[IS] = 30
    """
    shape = (10,20,30)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x,y,z] = (1+x)*1 + (1+y)*100 + (1+z)*10000

    if 0:
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    sys.stdout.write(" % 3d" % data[x,y,z])
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


def fake_3dimage_sct():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage()
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img


def test_transfo_null():
    dir_tmp = "."

    print("Null warping field")

    print(" Create some recognizable data")
    data = np.ones((7,8,9), order="F")
    path_src = "warp-src.nii"
    img_src = fake_image_sct_custom(data).save(path_src)

    print(" Create a null warping field")
    data = np.zeros((7,8,9,1,3), order="F")
    path_warp = "warp-field.nii"
    img_warp = fake_image_sct_custom(data)
    img_warp.header.set_intent('vector', (), '')
    img_warp.save(path_warp)

    print(" Apply")
    path_dst = "warp-dst.nii"
    xform = sct_apply_transfo.Transform(input_filename=path_src, fname_dest=path_src, list_warp=[path_warp],
                                        output_filename=path_dst)
    xform.apply()

    img_src2 = msct_image.Image(path_src)
    img_dst = msct_image.Image(path_dst)

    print(" Check that the source wasn't altered")
    assert (img_src.data == img_src2.data).all()

    assert img_src.orientation == img_dst.orientation
    assert img_src.data.shape == img_dst.data.shape

    print(" Check that the destination is identical to the source since the field was null")
    dat_src = img_src.data
    dat_dst = np.array(img_dst.data)
    for idx_slice, (slice_src, slice_dst) in enumerate(msct_image.SlicerMany((img_src, img_dst), msct_image.Slicer)):
        slice_src = np.array(slice_src)
        slice_dst = np.array(slice_dst)
        print(slice_src)
        print(slice_dst)
        assert np.allclose(slice_src, slice_dst)
    assert np.allclose(dat_src, dat_dst)



def test_transfo_figure_out_ants_frame_exhaustive():

    dir_tmp = "."
    all_orientations = msct_image.all_refspace_strings()
    #all_orientations = ("LPS", "LPI")

    print("Wondering which orientation is native to ANTs")

    working_orientations = [] # there can't be only one...

    for orientation in all_orientations:
        print(" Shifting +1,+1,+1 (in {})".format(orientation))

        path_src = "warp-{}-src.nii".format(orientation)
        img_src = fake_3dimage_sct().change_orientation(orientation).save(path_src)

        # Create warping field
        shape = tuple(list(img_src.data.shape) + [1,3])
        data = np.ones(shape, order="F")
        path_warp = "warp-{}-field.nii".format(orientation)
        img_warp = fake_image_sct_custom(data)
        img_warp.header.set_intent('vector', (), '')
        img_warp.change_orientation(orientation).save(path_warp)
        print(" Affine:\n{}".format(img_warp.header.get_best_affine()))

        path_dst = "warp-{}-dst.nii".format(orientation)
        xform = sct_apply_transfo.Transform(input_filename=path_src, fname_dest=path_src, list_warp=[path_warp],
                                            output_filename=path_dst)
        xform.apply()

        img_src2 = msct_image.Image(path_src)
        img_dst = msct_image.Image(path_dst)

        assert img_src.orientation == img_dst.orientation
        assert img_src.data.shape == img_dst.data.shape

        dat_src = img_src.data
        dat_dst = np.array(img_dst.data)

        value = 222
        aff_src = img_dst.header.get_best_affine()
        aff_dst = img_dst.header.get_best_affine()
        pt_src = np.array(np.unravel_index(np.argmin(np.abs(dat_src - value)), dat_src.shape))#, order="F"))
        pt_dst = np.array(np.unravel_index(np.argmin(np.abs(dat_dst - value)), dat_dst.shape))#, order="F"))
        print("Point %s -> %s" % (pt_src, pt_dst))

        pos_src = np.matmul(aff_src, np.hstack((pt_src, [1])).reshape((4,1)))
        pos_dst = np.matmul(aff_dst, np.hstack((pt_dst, [1])).reshape((4,1)))

        displacement = (pos_dst - pos_src).T[:3]
        print("Displacement (physical): %s" % (displacement))
        displacement = pt_dst - pt_src
        print("Displacement (logical): %s" % (displacement))

        assert dat_src.shape == dat_dst.shape

        if 0:
            for idx_slice in range(9):
                print(dat_src[...,idx_slice])
                print(dat_dst[...,idx_slice])
                print("")

        try:
            # Check same as before
            assert np.allclose(dat_dst[0,:,:], 0)
            assert np.allclose(dat_dst[:,0,:], 0)
            assert np.allclose(dat_dst[:,:,0], 0)
            assert np.allclose(dat_src[:-1,:-1,:-1], dat_dst[1:,1:,1:])
            working_orientations.append(orientation)
        except AssertionError as e:
            continue
            print("\x1B[31;1m Failed in {}\x1B[0m".format(orientation))
            for idx_slice in range(shape[2]):
                print(dat_src[...,idx_slice])
                print(dat_dst[...,idx_slice])
                print("")

    print("-> Working orientation: {}".format(" ".join(working_orientations)))


def test_transfo_exhaustive_wrt_orientations():

    dir_tmp = "."

    print("Figuring out which orientations work without workaround")

    all_orientations = msct_image.all_refspace_strings()

    orientations_ok = []
    orientations_ng = []
    orientations_dk = []

    for orientation in all_orientations:
        shift = np.array([1,2,3])
        shift_wanted = shift.copy()
        shift[2] *= -1 # ANTs / ITK reference frame is LPS, ours is LPI
        # (see docs or test_transfo_figure_out_ants_frame_exhaustive())

        print(" Shifting {} in {}".format(shift_wanted, orientation))

        path_src = "warp-{}-src.nii".format(orientation)
        img_src = fake_3dimage_sct().change_orientation(orientation).save(path_src)

        path_ref = path_src
        img_ref = img_src

        # Create warping field
        shape = tuple(list(img_src.data.shape) + [1,3])
        data = np.zeros(shape, order="F")
        data[:,:,:,0] = shift

        path_warp = "warp-{}-field.nii".format(orientation)
        img_warp = fake_image_sct_custom(data)
        img_warp.header.set_intent('vector', (), '')
        img_warp.change_orientation(orientation).save(path_warp)
        #print(" Affine:\n{}".format(img_warp.header.get_best_affine()))

        path_dst = "warp-{}-dst.nii".format(orientation)
        xform = sct_apply_transfo.Transform(input_filename=path_src, fname_dest=path_src, list_warp=[path_warp],
                                            output_filename=path_dst)
        xform.apply()

        img_src2 = msct_image.Image(path_src)
        img_dst = msct_image.Image(path_dst)

        assert img_src.orientation == img_dst.orientation
        assert img_src.data.shape == img_dst.data.shape

        dat_src = img_src.data
        dat_dst = np.array(img_dst.data)

        value = 50505
        aff_src = img_src.header.get_best_affine()
        aff_dst = img_dst.header.get_best_affine()

        pt_src = np.argwhere(dat_src == value)[0]
        try:
            pt_dst = np.argwhere(dat_dst == value)[0]
            1/0
        except:
            min_ = np.round(np.min(np.abs(dat_dst - value)), 2)
            pt_dst = np.array(np.unravel_index(np.argmin(np.abs(dat_dst - value)), dat_dst.shape))#, order="F"))

        print(" Point %s -> %s (%s) %s" % (pt_src, pt_dst, dat_dst[tuple(pt_dst)], min_))
        if min_ != 0:
            orientations_dk.append(orientation)
            continue

        pos_src = np.matmul(aff_src, np.hstack((pt_src, [1])).reshape((4,1)))
        pos_dst = np.matmul(aff_dst, np.hstack((pt_dst, [1])).reshape((4,1)))

        displacement = (pos_dst - pos_src).reshape((-1))[:3]
        displacement_log = pt_dst - pt_src
        #print(" Displacement (logical): %s" % (displacement_log))
        if not np.allclose(displacement, shift_wanted):
            orientations_ng.append(orientation)
            print(" \x1B[31;1mDisplacement (physical): %s\x1B[0m" % (displacement))
        else:
            orientations_ok.append(orientation)
            print(" Displacement (physical): %s" % (displacement))
        print("")

    print("Orientations OK: {}".format(" ".join(orientations_ok)))
    print("Orientations NG: {}".format(" ".join(orientations_ng)))
    print("Orientations DK: {}".format(" ".join(orientations_dk)))


def notest_transfo_more_exhaustive_wrt_orientations():

    dir_tmp = "."

    print("Figuring out which orientations work without workaround")

    all_orientations = msct_image.all_refspace_strings()

    orientations_ok = []
    orientations_ng = []
    orientations_dk = []

    for orientation_src in all_orientations:
        for orientation_ref in all_orientations:
            shift = np.array([1,2,3])
            shift_wanted = shift.copy()
            shift[2] *= -1 # ANTs / ITK reference frame is LPS, ours is LPI
            # (see docs or test_transfo_figure_out_ants_frame_exhaustive())

            print(" Shifting {} in {} ref {}".format(shift_wanted, orientation_src, orientation_ref))

            path_src = "warp2-{}.nii".format(orientation_src)
            img_src = fake_3dimage_sct().change_orientation(orientation_src).save(path_src)

            path_ref = "warp2-{}.nii".format(orientation_ref)
            img_ref = fake_3dimage_sct().change_orientation(orientation_ref).save(path_ref)


            # Create warping field
            shape = tuple(list(img_src.data.shape) + [1,3])
            data = np.zeros(shape, order="F")
            data[:,:,:,0] = shift

            path_warp = "warp-{}-{}-field.nii".format(orientation_src, orientation_ref)
            img_warp = fake_image_sct_custom(data)
            img_warp.header.set_intent('vector', (), '')
            img_warp.change_orientation(orientation_ref).save(path_warp)
            #print(" Affine:\n{}".format(img_warp.header.get_best_affine()))

            path_dst = "warp-{}-{}-dst.nii".format(orientation_src, orientation_ref)
            xform = sct_apply_transfo.Transform(input_filename=path_src, fname_dest=path_src, list_warp=[path_warp],
                                                output_filename=path_dst)
            xform.apply()

            img_src2 = msct_image.Image(path_src)
            img_dst = msct_image.Image(path_dst)

            assert img_ref.orientation == img_dst.orientation
            assert img_ref.data.shape == img_dst.data.shape

            dat_src = img_src.data
            dat_dst = np.array(img_dst.data)

            value = 50505
            aff_src = img_src.header.get_best_affine()
            aff_dst = img_dst.header.get_best_affine()

            pt_src = np.argwhere(dat_src == value)[0]
            try:
                pt_dst = np.argwhere(dat_dst == value)[0]
                1/0
            except:
                # Work around numerical inaccuracy, that is somehow introduced by ANTs
                min_ = np.round(np.min(np.abs(dat_dst - value)), 1)
                pt_dst = np.array(np.unravel_index(np.argmin(np.abs(dat_dst - value)), dat_dst.shape))#, order="F"))

            print(" Point %s -> %s (%s) %s" % (pt_src, pt_dst, dat_dst[tuple(pt_dst)], min_))
            if min_ != 0:
                orientations_dk.append((orientation_src, orientation_ref))
                continue

            pos_src = np.matmul(aff_src, np.hstack((pt_src, [1])).reshape((4,1)))
            pos_dst = np.matmul(aff_dst, np.hstack((pt_dst, [1])).reshape((4,1)))

            displacement = (pos_dst - pos_src).reshape((-1))[:3]
            displacement_log = pt_dst - pt_src
            #print(" Displacement (logical): %s" % (displacement_log))
            if not np.allclose(displacement, shift_wanted):
                orientations_ng.append((orientation_src, orientation_ref))
                print(" \x1B[31;1mDisplacement (physical): %s\x1B[0m" % (displacement))
            else:
                orientations_ok.append((orientation_src, orientation_ref))
                print(" Displacement (physical): %s" % (displacement))
            print("")

    def ori_str(x):
        return " ".join(["{}->{}".format(x,y) for (x,y) in x])

    print("Orientations OK: {}".format(ori_str(orientations_ok)))
    print("Orientations NG: {}".format(ori_str(orientations_ng)))
    print("Orientations DK: {}".format(ori_str(orientations_dk)))



def test_transfo_skip_pix2phys():
    # "Recipe" useful if you want to skip pix2phys, which is *not* a good idea

    dir_tmp = "."

    print("Achieving a shifting of +1,+1,+1 (in LPI)")

    print(" Create data")

    path_src = "warp-src.nii"
    img_src = fake_3dimage_sct().save(path_src)


    print(" Create a warping field" \
"""
  Now we want to shift things by (+1,+1,+1), meaning
  that a destination voxel at position (x_1, y_1, z_1)
  contains the stuff that was in the source voxel at position
  (x_1-1, y_1-1, z_1-1).
""")
    shape = tuple(list(img_src.data.shape) + [1,3])

    data = np.ones(shape, order="F")
    data[...,2] *= -1 # invert Z so that the result is what we expect
    # See test_transfo_exhaustive_wrt_orientations()

    path_warp = "warp-field111.nii"
    img_warp = fake_image_sct_custom(data)
    img_warp.header.set_intent('vector', (), '')
    img_warp.save(path_warp)

    path_dst = "warp-dst111.nii"
    xform = sct_apply_transfo.Transform(input_filename=path_src, fname_dest=path_src, list_warp=[path_warp],
                                        output_filename=path_dst)
    xform.apply()

    img_src2 = msct_image.Image(path_src)
    img_dst = msct_image.Image(path_dst)

    print(" Check that the source wasn't altered")
    assert (img_src.data == img_src2.data).all()

    assert img_src.orientation == img_dst.orientation
    assert img_src.data.shape == img_dst.data.shape

    print(" Check the contents")
    dat_src = img_src.data
    dat_dst = np.array(img_dst.data)

    if 0:
        for idx_slice in range(shape[2]):
            print("")
            print(dat_src[...,idx_slice])
            print(dat_dst[...,idx_slice])

    # The volume should be shifted by 1 and the "lowest faces" of the destination
    # should be empty
    assert np.allclose(dat_dst[0,:,:], 0)
    assert np.allclose(dat_dst[:,0,:], 0)
    assert np.allclose(dat_dst[:,:,0], 0)
    assert np.allclose(dat_src[:-1,:-1,:-1], dat_dst[1:,1:,1:])




