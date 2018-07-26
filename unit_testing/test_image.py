#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for Image stuff

import sys, io, os, time, itertools

import pytest

import numpy as np
import nibabel
import nibabel.orientations

import sct_utils as sct
import msct_image
import sct_image

@pytest.fixture(scope="session")
def image_paths():
    ret = []
    sct_dir = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(sct_dir, "data")
    for cwd, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith((".nii.gz", ".nii")):
                path = os.path.join(cwd, file)
                ret.append(path)
    return ret



def fake_3dimage_custom(data):
    """
    :return: a Nifti1Image (3D) in RAS+ space
    """
    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)


def fake_3dimage_sct_custom(data):
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage_custom(data)
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img


@pytest.fixture(scope="session")
def fake_3dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 7
    - shape[PA] = 8
    - shape[IS] = 9
    """
    shape = (7,8,9)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x,y,z] = (1+x)*1 + (1+y)*10 + (1+z)*100

    if 0:
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    sys.stdout.write(" % 3d" % data[x,y,z])
                sys.stdout.write("\n")
            sys.stdout.write("\n")

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)

@pytest.fixture(scope="session")
def fake_3dimage2():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 1
    - shape[PA] = 2
    - shape[IS] = 3
    """
    shape = (1,2,3)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for z in range(shape[2]):
        for y in range(shape[1]):
            for x in range(shape[0]):
                data[x,y,z] = (1+x)*1 + (1+y)*10 + (1+z)*100

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)

@pytest.fixture(scope="session")
def fake_4dimage():
    """
    :return: a Nifti1Image (3D) in RAS+ space

    Following characteristics:

    - shape[LR] = 2
    - shape[PA] = 3
    - shape[IS] = 4
    - shape[t] = 5
    """
    shape = (2,3,4,5)
    data = np.zeros(shape, dtype=np.float32, order="F")

    for t in range(shape[3]):
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    data[x,y,z,t] = (1+x)*1 + (1+y)*10 + (1+z)*100 + (1+t)*1000

    affine = np.eye(4)
    return nibabel.nifti1.Nifti1Image(data, affine)

@pytest.fixture(scope="session")
def fake_4dimage_sct():
    """
    :return: an Image (4D) in RAS+ (aka SCT LPI) space
    """
    i = fake_4dimage()
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img

@pytest.fixture(scope="session")
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

@pytest.fixture(scope="session")
def fake_3dimage_sct2():
    """
    :return: an Image (3D) in RAS+ (aka SCT LPI) space
    """
    i = fake_3dimage2()
    img = msct_image.Image(i.get_data(), hdr=i.header,
     orientation="LPI",
     dim=i.header.get_data_shape(),
    )
    return img


def test_slicer(fake_3dimage_sct, fake_3dimage_sct2):
    im3d = fake_3dimage_sct
    slicer = msct_image.Slicer(im3d, "IS")
    assert slicer.direction == +1
    assert slicer.nb_slices == 9
    if 0:
        for im2d in slicer:
            print(im2d)

    assert 100 < np.mean(slicer[0]) < 200

    slicer = msct_image.Slicer(im3d, "SI")
    assert slicer.direction == -1
    assert slicer.nb_slices == 9

    if 0:
        for im2d in slicer:
            print(im2d)

    assert 900 < np.mean(slicer[0]) < 1000

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LI")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "L")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "Lx")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(im3d, "LA")

    im3d2 = fake_3dimage_sct.copy()
    im3d2.data += 1000
    slicer = msct_image.Slicer((im3d, im3d2), "IS")
    for idx_slice, (im2d_a, im2d_b) in enumerate(slicer):
        assert np.all(im2d_b == im2d_a + 1000)

    im3d2 = fake_3dimage_sct2
    with pytest.raises(ValueError):
        slicer = msct_image.Slicer((im3d, im3d2), "IS")

    with pytest.raises(ValueError):
        slicer = msct_image.Slicer(1)


def test_nibabel(fake_3dimage):
    img = fake_3dimage
    print(img.header)


def test_nibabel_reorient(fake_3dimage):
    """
    nibabel can reorient since recent versions,
    it's not the best idea to use that feature yet.
    """
    return
    src = fake_3dimage
    ornt = (
     [1, 1],
     [2, 1],
     [0,-1],
    )
    dst = src.as_reoriented(ornt)
    print(dst.header)



def test_change_orientation(fake_3dimage_sct, fake_3dimage_sct_vis):

    path_tmp = sct.tmp_create(basename="test_reorient")
    path_tmp = "."

    print("Spot-checking that physical coordinates don't change")
    for shape_is in (1,2,3):
        shape = (1,1,shape_is)
        print("Simple image with shape {}".format(shape))

        data = np.ones(shape, order="F")
        data[:,:,shape_is-1] += 1
        im_src = fake_3dimage_sct_custom(data)
        im_dst = msct_image.change_orientation(im_src, "ASR")
        # Basic check
        assert im_dst.orientation == "ASR"
        # Basic data check
        assert im_dst.data.mean() == im_src.data.mean()
        # Basic header check: check that the same voxel
        # remains at the same physical position
        aff_src = im_src.header.get_best_affine()
        aff_dst = im_dst.header.get_best_affine()

        # Take the extremities "a" & "z"...
        # consider the original LPI position
        pta_src = np.array([[0,0,0,1]]).T
        ptz_src = np.array([[0,0,shape_is-1,1]]).T
        # and the position in ASR
        pta_dst = np.array([[0,shape_is-1,0,1]]).T
        ptz_dst = np.array([[0,0,0,1]]).T
        # The physical positions should be:
        posa_src = np.matmul(aff_src, pta_src)
        posa_dst = np.matmul(aff_dst, pta_dst)
        print("A at src {}".format(posa_src.T))
        print("A at dst {}".format(posa_dst.T))
        posz_src = np.matmul(aff_src, ptz_src)
        posz_dst = np.matmul(aff_dst, ptz_dst)
        # and they should be equal
        assert (posa_src == posa_dst).all()
        assert (posz_src == posz_dst).all()
        fn = "".join(str(x) for x in im_src.data.shape)
        im_src.save("{}-src.nii".format(fn))
        im_dst.save("{}-dst.nii".format(fn))


    print("More checking that physical coordinates don't change")
    if 1:
        shape = (7,8,9)
        print("Simple image with shape {}".format(shape))

        data = np.ones(shape, order="F") * 10
        data[4,4,4] = 4
        data[3,3,3] = 3
        data[0,0,0] = 0
        values = (0,3,4)

        im_ref = fake_3dimage_sct_custom(data)
        im_ref.header.set_xyzt_units("mm", "msec")

        import scipy.linalg

        def rand_rot():
            q, _ = scipy.linalg.qr(np.random.randn(3, 3))
            if scipy.linalg.det(q) < 0:
                q[:, 0] = -q[:, 0]
            return q

        affine = im_ref.header.get_best_affine()
        affine[:3,:3] = rand_rot()
        affine[3,:3] = 0.0
        affine[:3,3] = np.random.random((3))
        affine[3,3] = 1.0

        affine[0,0] *= 2
        im_ref.header.set_sform(affine, code='scanner')

        orientations = msct_image.all_refspace_strings()
        for ori_src in orientations:
            for ori_dst in orientations:
                print("{} -> {}".format(ori_src, ori_dst))
                im_src = msct_image.change_orientation(im_ref, ori_src)
                im_dst = msct_image.change_orientation(im_src, ori_dst)

                assert im_src.orientation == ori_src
                assert im_dst.orientation == ori_dst
                assert im_dst.data.mean() == im_src.data.mean()

                # Basic header check: check that the same voxel
                # remains at the same physical position
                aff_src = im_src.header.get_best_affine()
                aff_dst = im_dst.header.get_best_affine()

                data_src = np.array(im_src.data)
                data_dst = np.array(im_dst.data)

                for value in values:
                    pt_src = np.argwhere(data_src == value)[0]
                    pt_dst = np.argwhere(data_dst == value)[0]

                    pos_src = np.matmul(aff_src, np.hstack((pt_src, [1])).reshape((4,1)))
                    pos_dst = np.matmul(aff_dst, np.hstack((pt_dst, [1])).reshape((4,1)))
                    if 0:
                        print("P at src {}".format(pos_src.T))
                        print("P at dst {}".format(pos_dst.T))
                    assert np.allclose(pos_src, pos_dst, atol=1e-3)

    im_src = fake_3dimage_sct.copy()
    im_src.save(os.path.join(path_tmp, "src.nii"), mutable=True)

    print(im_src.orientation, im_src.data.shape)

    def orient2shape(orient):
        # test-data-specific thing
        letter2dim = dict(
         L=7,
         R=7,
         A=8,
         P=8,
         I=9,
         S=9,
        )
        return tuple([letter2dim[x] for x in orient])

    orientation = im_src.orientation # LPI
    assert im_src.header.get_best_affine()[:3,3].tolist() == [0,0,0]
    im_dst = msct_image.change_orientation(im_src, "RPI")
    print(im_dst.orientation, im_dst.data.shape)
    assert im_dst.data.shape == orient2shape("RPI")
    assert im_dst.header.get_best_affine()[:3,3].tolist() == [7-1,0,0]

    # spot check
    orientation = im_src.orientation # LPI
    im_dst = msct_image.change_orientation(im_src, "IRP")
    print(im_dst.orientation, im_dst.data.shape)
    assert im_dst.data.shape == orient2shape("IRP")

    # to & fro
    im_dst2 = msct_image.change_orientation(im_dst, orientation)
    print(im_dst2.orientation, im_dst2.data.shape)
    assert im_dst2.orientation == im_src.orientation
    assert im_dst2.data.shape == orient2shape(orientation)
    assert (im_dst2.data == im_src.data).all()
    assert np.allclose(im_src.header.get_best_affine(), im_dst2.header.get_best_affine())


    # copy
    im_dst = im_src.copy().change_orientation("IRP")
    assert im_dst.data.shape == orient2shape("IRP")
    print(im_dst.orientation, im_dst.data.shape)


    print("Testing orientation persistence")
    img = im_src.copy()
    orientation = img.orientation
    fn = os.path.join(path_tmp, "pouet.nii")
    img.change_orientation("PIR").save(fn)
    assert img.data.shape == orient2shape("PIR")
    img = msct_image.Image(fn)
    assert img.orientation == "PIR"
    assert img.data.shape == orient2shape("PIR")
    print(img.orientation, img.data.shape)

    possibilities = msct_image.all_refspace_strings()
    for orientation in possibilities:
        dst = msct_image.change_orientation(im_src, orientation)
        assert orientation == dst.orientation


def test_change_nd_orientation(fake_4dimage_sct):
    import sct_image

    im_src = fake_4dimage_sct.copy()
    path_tmp = sct.tmp_create(basename="test_reorient")
    im_src.save(os.path.join(path_tmp, "src.nii"), mutable=True)

    print(im_src.orientation, im_src.data.shape)

    def orient2shape(orient):
        # test-data-specific thing
        letter2dim = dict(
         L=2,
         R=2,
         A=3,
         P=3,
         I=4,
         S=4,
        )
        return tuple([letter2dim[x] for x in orient] + [5])

    orientation = im_src.orientation
    assert orientation == "LPI"
    assert im_src.header.get_best_affine()[:3,3].tolist() == [0,0,0]

    im_dst = msct_image.change_orientation(im_src, "RPI")
    assert im_dst.orientation == "RPI"
    assert im_dst.data.shape == orient2shape("RPI")
    assert im_dst.header.get_best_affine()[:3,3].tolist() == [2-1,0,0]



def test_crop(fake_3dimage_sct):


    im_src = fake_3dimage_sct.copy()

    crop_spec = dict(((0, (1,3)),(1, (2,4)),(2, (3,5))))
    print(crop_spec)

    im_dst = msct_image.spatial_crop(im_src, crop_spec)

    print("Check shape")
    assert im_dst.data.shape == (3,3,3)
    print("Check world pos")
    aff_src = im_src.header.get_best_affine()
    aff_dst = im_dst.header.get_best_affine()
    pos_src = np.matmul(aff_src, np.array([[1,2,3,1]]).T)
    pos_dst = np.matmul(aff_dst, np.array([[0,0,0,1]]).T)
    assert (pos_src == pos_dst).all()


def test_change_shape(fake_3dimage_sct):

    # Add dimension
    im_src = fake_3dimage_sct
    shape = tuple(list(im_src.data.shape) + [1])
    im_dst = msct_image.change_shape(im_src, shape)
    path_tmp = sct.tmp_create(basename="test_reshape")
    src_path = os.path.join(path_tmp, "src.nii")
    dst_path = os.path.join(path_tmp, "dst.nii")
    im_src.save(src_path)
    im_dst.save(dst_path)
    im_src = msct_image.Image(src_path)
    im_dst = msct_image.Image(dst_path)
    assert im_dst.data.shape == shape

    data_src = im_src.data
    data_dst = im_dst.data

    assert (data_dst.reshape(data_src.shape) == data_src).all()

    # Remove dimension
    im_dst = im_dst.change_shape(im_src.data.shape)
    assert im_dst.data.shape == im_src.data.shape



