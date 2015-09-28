#!/usr/bin/env python
#########################################################################################
#
# Image class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Augustin Roux, Benjamin De Leener
# Modified: 2015-02-20
#
# About the license: see the file LICENSE.TXT
#########################################################################################

# TODO: update function to reflect the new get_dimension

class Image(object):
    """

    """
    def __init__(self, param=None, hdr=None, orientation=None, absolutepath="", dim=None, verbose=1):
        from numpy import zeros, ndarray, generic
        from sct_utils import extract_fname
        from nibabel import AnalyzeHeader

        # initialization of all parameters
        self.data = None
        self.orientation = None
        self.absolutepath = ""
        self.path = ""
        self.file_name = ""
        self.ext = ""
        self.dim = None

        if hdr is None:
            hdr = self.hdr = AnalyzeHeader()  # an empty header
        else:
            self.hdr = hdr

        self.verbose = verbose

        # load an image from file
        if type(param) is str:
            self.loadFromPath(param, verbose)
        # copy constructor
        elif isinstance(param, type(self)):
            self.copy(param)
        # create an empty image (full of zero) of dimension [dim]. dim must be [x,y,z] or (x,y,z). No header.
        elif type(param) is list:
            self.data = zeros(param)
            self.dim = param
            self.hdr = hdr
            self.orientation = orientation
            self.absolutepath = absolutepath
            self.path, self.file_name, self.ext = extract_fname(absolutepath)
        # create a copy of im_ref
        elif isinstance(param, (ndarray, generic)):
            self.data = param
            self.dim = dim
            self.hdr = hdr
            self.orientation = orientation
            self.absolutepath = absolutepath
            self.path, self.file_name, self.ext = extract_fname(absolutepath)
        else:
            raise TypeError('Image constructor takes at least one argument.')

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(self.data, memo), deepcopy(self.hdr, memo), deepcopy(self.orientation, memo), deepcopy(self.absolutepath, memo), deepcopy(self.dim, memo))

    def copy(self, image=None):
        from copy import deepcopy
        from sct_utils import extract_fname
        if image is not None:
            self.data = deepcopy(image.data)
            self.dim = deepcopy(image.dim)
            self.hdr = deepcopy(image.hdr)
            self.orientation = deepcopy(image.orientation)
            self.absolutepath = deepcopy(image.absolutepath)
            self.path, self.file_name, self.ext = extract_fname(self.absolutepath)
        else:
            return deepcopy(self)

    def loadFromPath(self, path, verbose):
        """
        This function load an image from an absolute path using nibabel library
        :param path: path of the file from which the image will be loaded
        :return:
        """
        from nibabel import load, spatialimages
        from sct_utils import check_file_exist, printv, extract_fname
        from sct_orientation import get_orientation

        # check_file_exist(path, verbose=verbose)
        im_file = None
        try:
            im_file = load(path)
        except spatialimages.ImageFileError:
            printv('Error: make sure ' + path + ' is an image.', 1, 'error')
        self.orientation = get_orientation(path)
        self.data = im_file.get_data()
        self.hdr = im_file.get_header()
        self.absolutepath = path
        self.path, self.file_name, self.ext = extract_fname(path)
        self.dim = get_dimension(im_file)
        # nx, ny, nz, nt, px, py, pz, pt = get_dimension(path)
        # self.dim = [nx, ny, nz]


    def setFileName(self, filename):
        """
        :param filename: file name with extension
        :return:
        """
        from sct_utils import extract_fname
        self.absolutepath = filename
        self.path, self.file_name, self.ext = extract_fname(filename)

    def changeType(self, type=''):
        """
        Change the voxel type of the image
        :param type:    if not set, the image is saved in standard type
                        if 'minimize', image space is minimize
                        if 'minimize_int', image space is minimize and values are approximated to integers
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
        :return:
        """
        from numpy import uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64

        if type == '':
            type = self.hdr.get_data_dtype()

        if type == 'minimize' or type == 'minimize_int':
            from numpy import nanmax, nanmin
            # compute max value in the image and choose the best pixel type to represent all the pixels within smallest memory space
            # warning: does not take intensity resolution into account, neither complex voxels
            max_vox = nanmax(self.data)
            min_vox = nanmin(self.data)

            # check if voxel values are real or integer
            isInteger = True
            if type == 'minimize':
                for vox in self.data.flatten():
                    if int(vox) != vox:
                        isInteger = False
                        break

            if isInteger:
                from numpy import iinfo, uint8, uint16, uint32, uint64
                if min_vox >= 0:  # unsigned
                    if max_vox <= iinfo(uint8).max:
                        type = 'uint8'
                    elif max_vox <= iinfo(uint16):
                        type = 'uint16'
                    elif max_vox <= iinfo(uint32).max:
                        type = 'uint32'
                    elif max_vox <= iinfo(uint64).max:
                        type = 'uint64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
                else:
                    if max_vox <= iinfo(int8).max and min_vox >= iinfo(int8).min:
                        type = 'int8'
                    elif max_vox <= iinfo(int16).max and min_vox >= iinfo(int16).min:
                        type = 'int16'
                    elif max_vox <= iinfo(int32).max and min_vox >= iinfo(int32).min:
                        type = 'int32'
                    elif max_vox <= iinfo(int64).max and min_vox >= iinfo(int64).min:
                        type = 'int64'
                    else:
                        raise ValueError("Maximum value of the image is to big to be represented.")
            else:
                from numpy import finfo, float32, float64
                # if max_vox <= np.finfo(np.float16).max and min_vox >= np.finfo(np.float16).min:
                #    type = 'np.float16' # not supported by nibabel
                if max_vox <= finfo(float32).max and min_vox >= finfo(float32).min:
                    type = 'float32'
                elif max_vox <= finfo(float64).max and min_vox >= finfo(float64).min:
                    type = 'float64'

        # print "The image has been set to "+type+" (previously "+str(self.hdr.get_data_dtype())+")"
        # change type of data in both numpy array and nifti header
        type_build = eval(type)
        self.data = type_build(self.data)
        self.hdr.set_data_dtype(type)

    def save(self, type=''):
        """
        Write an image in a nifti file
        :param type:    if not set, the image is saved in the same type as input data
                        if 'minimize', image space is minimize
                        (2, 'uint8', np.uint8, "NIFTI_TYPE_UINT8"),
                        (4, 'int16', np.int16, "NIFTI_TYPE_INT16"),
                        (8, 'int32', np.int32, "NIFTI_TYPE_INT32"),
                        (16, 'float32', np.float32, "NIFTI_TYPE_FLOAT32"),
                        (32, 'complex64', np.complex64, "NIFTI_TYPE_COMPLEX64"),
                        (64, 'float64', np.float64, "NIFTI_TYPE_FLOAT64"),
                        (256, 'int8', np.int8, "NIFTI_TYPE_INT8"),
                        (512, 'uint16', np.uint16, "NIFTI_TYPE_UINT16"),
                        (768, 'uint32', np.uint32, "NIFTI_TYPE_UINT32"),
                        (1024,'int64', np.int64, "NIFTI_TYPE_INT64"),
                        (1280, 'uint64', np.uint64, "NIFTI_TYPE_UINT64"),
                        (1536, 'float128', _float128t, "NIFTI_TYPE_FLOAT128"),
                        (1792, 'complex128', np.complex128, "NIFTI_TYPE_COMPLEX128"),
                        (2048, 'complex256', _complex256t, "NIFTI_TYPE_COMPLEX256"),
        """
        from nibabel import Nifti1Image, save
        from sct_utils import printv
        from numpy import squeeze
        from os import path, remove
        # remove singleton
        self.data = squeeze(self.data)
        if type != '':
            self.changeType(type)
        # update header
        if self.hdr:
            self.hdr.set_data_shape(self.data.shape)
        img = Nifti1Image(self.data, None, self.hdr)
        fname_out = self.path + self.file_name + self.ext
        if path.isfile(fname_out):
            printv('WARNING: File '+fname_out+' already exists. Deleting it.', 1, 'warning')
            remove(fname_out)
        # save file
        save(img, fname_out)

    # flatten the array in a single dimension vector, its shape will be (d, 1) compared to the flatten built in method
    # which would have returned (d,)
    def flatten(self):
        # return self.data.flatten().reshape(self.data.flatten().shape[0], 1)
        return self.data.flatten()

    # return a list of the image slices flattened
    def slices(self):
        slices = []
        for slc in self.data:
            slices.append(slc.flatten())
        return slices

    def getNonZeroCoordinates(self, sorting=None, reverse_coord=False, coordValue=False):
        """
        This function return all the non-zero coordinates that the image contains.
        Coordinate list can also be sorted by x, y, z, or the value with the parameter sorting='x', sorting='y', sorting='z' or sorting='value'
        If reverse_coord is True, coordinate are sorted from larger to smaller.
        """
        from msct_types import Coordinate
        from sct_utils import printv
        n_dim = 1
        if self.dim[3] == 1:
            n_dim = 3
        else:
            n_dim = 4
        if self.dim[2] == 1:
            n_dim = 2

        try:
            if n_dim == 3:
                X, Y, Z = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            elif n_dim == 2:
                X, Y = (self.data > 0).nonzero()
                list_coordinates = [Coordinate([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        except Exception, e:
            print 'ERROR', e
            printv('ERROR: Exception ' + str(e) + ' caught while geting non Zeros coordinates', 1, 'error')

        if coordValue:
            from msct_types import CoordinateValue
            if n_dim == 3:
                list_coordinates = [CoordinateValue([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            else:
                list_coordinates = [CoordinateValue([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        else:
            from msct_types import Coordinate
            if n_dim == 3:
                list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
            else:
                list_coordinates = [Coordinate([X[i], Y[i], self.data[X[i], Y[i]]]) for i in range(0, len(X))]
        if sorting is not None:
            if reverse_coord not in [True, False]:
                raise ValueError('reverse_coord parameter must be a boolean')

            if sorting == 'x':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.x, reverse=reverse_coord)
            elif sorting == 'y':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.y, reverse=reverse_coord)
            elif sorting == 'z':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.z, reverse=reverse_coord)
            elif sorting == 'value':
                list_coordinates = sorted(list_coordinates, key=lambda obj: obj.value, reverse=reverse_coord)
            else:
                raise ValueError("sorting parameter must be either 'x', 'y', 'z' or 'value'")

        return list_coordinates

    # crop the image in order to keep only voxels in the mask, therefore the mask's slices must be squares or
    # rectangles of the same size
    # orientation must be IRP to be able to go trough slices as first dimension
    # This method is called in sct_crop_over_mask script
    def crop_from_square_mask(self, mask, save=True):
        from numpy import asarray, zeros

        data_array = self.data
        data_mask = mask.data
        assert self.orientation == 'IRP'
        assert mask.orientation == 'IRP'

        print 'ORIGINAL SHAPE: ', data_array.shape, '   ==   ', data_mask.shape
        # if the image to crop is smaller than the mask in total, we assume the image was centered and add a padding to fit the mask's shape
        if data_array.shape != data_mask.shape:
            old_data_array = data_array
            pad_1 = int((data_mask.shape[1] - old_data_array.shape[1])/2 + 1)
            pad_2 = int((data_mask.shape[2] - old_data_array.shape[2])/2 + 1)

            data_array = zeros(data_mask.shape)
            for n_slice, data_slice in enumerate(data_array):
                data_slice[pad_1:pad_1+old_data_array.shape[1], pad_2:pad_2+old_data_array.shape[2]] = old_data_array[n_slice]
            '''
            for n_slice, data_slice in enumerate(data_array):
                n_row_old_data_array = 0
                for row in data_slice[pad_2:-pad_2-1]:
                    row[pad_1:pad_1 + old_data_array.shape[1]] = old_data_array[n_slice, n_row_old_data_array]
                    n_row_old_data_array += 1
            '''
            self.data = data_array
            if save:
                self.file_name += '_resized'
                self.save()

        data_array = asarray(data_array)
        data_mask = asarray(data_mask)
        new_data = []
        buffer = []
        buffer_mask = []

        if len(data_array.shape) == 3:
            for n_slice, mask_slice in enumerate(data_mask):
                for n_row, row in enumerate(mask_slice):
                    if sum(row) > 0:  # and n_row<=data_array.shape[1] and n_slice<=data_array.shape[0]:
                        buffer_mask.append(row)
                        buffer.append(data_array[n_slice][n_row])

                new_slice_mask = asarray(buffer_mask).T
                new_slice = asarray(buffer).T
                buffer = []
                for n_row, row in enumerate(new_slice_mask):
                    if sum(row) != 0:
                        buffer.append(new_slice[n_row])
                new_slice = asarray(buffer).T
                buffer_mask = []
                buffer = []
                new_data.append(new_slice)

        elif len(data_array.shape) == 2:
            for n_row, row in enumerate(data_mask):
                if sum(row) > 0:  # and n_row<=data_array.shape[1] and n_slice<=data_array.shape[0]:
                    buffer_mask.append(row)
                    buffer.append(data_array[n_row])

            new_slice_mask = asarray(buffer_mask).T
            new_slice = asarray(buffer).T
            buffer = []
            for n_row, row in enumerate(new_slice_mask):
                if sum(row) != 0:
                    buffer.append(new_slice[n_row])
            new_data = asarray(buffer).T
            buffer_mask = []
            buffer = []

        new_data = asarray(new_data)
        # print data_mask
        self.data = new_data
        self.dim = self.data.shape


    # crop the image in order to keep only voxels in the mask
    # doesn't change the image dimension
    # This method is called in sct_crop_over_mask script
    def crop_from_mask(self, mask):
        from numpy import asarray, einsum
        data_array = self.data
        data_mask = mask.data
        assert data_array.shape == data_mask.shape
        array = asarray(data_array)
        data_mask = asarray(data_mask)

        #Element-wise matrix multiplication:
        new_data = None
        if len(data_array.shape) == 3:
            new_data = einsum('ijk,ijk->ijk', data_mask, array)
        elif len(data_array.shape) == 2:
            new_data = einsum('ij,ij->ij', data_mask, array)
        self.data = new_data

    def denoise_ornlm(self):
        from commands import getstatusoutput
        from sys import path
        # append python path for importing module
        # N.B. PYTHONPATH variable should take care of it, but this is only used for Travis.
        status, path_sct = getstatusoutput('echo $SCT_DIR')
        path.append(path_sct + '/external/denoise/ornlm')
        from ornlm import ornlm
        import numpy as np
        dat = self.data.astype(np.float64)
        denoised = np.array(ornlm(dat, 3, 1, np.max(dat)*0.01))
        self.file_name += '_denoised'
        self.data = denoised

    def invert(self):
        self.data = self.data.max() - self.data
        return self

    def change_orientation(self, orientation='RPI', inversion_orient=False):
        """
        This function changes the orientation of the data by swapping the image axis.
        Warning: the nifti image header is not changed!!!
        :param orientation: string of three character representing the new orientation (ex: AIL, default: RPI)
               inversion_orient: boolean. If True, the data change to match the orientation in the header, based on the orientation provided as the argument orientation.
        :return:
        """
        opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

        if self.orientation is None:
            from sct_orientation import get_orientation
            self.orientation = get_orientation(self.file_name)
        # get orientation to return at the end of function
        raw_orientation = self.orientation

        if inversion_orient:
            temp_orientation = self.orientation
            self.orientation = orientation
            orientation = temp_orientation

        # change the orientation of the image
        perm = [0, 1, 2]
        inversion = [1, 1, 1]
        for i, character in enumerate(self.orientation):
            try:
                perm[i] = orientation.index(character)
            except ValueError:
                perm[i] = orientation.index(opposite_character[character])
                inversion[i] = -1

        # axes inversion
        self.data = self.data[::inversion[0], ::inversion[1], ::inversion[2]]

        # axes manipulations
        from numpy import swapaxes

        if perm == [1, 0, 2]:
            self.data = swapaxes(self.data, 0, 1)
        elif perm == [2, 1, 0]:
            self.data = swapaxes(self.data, 0, 2)
        elif perm == [0, 2, 1]:
            self.data = swapaxes(self.data, 1, 2)
        elif perm == [2, 0, 1]:
            self.data = swapaxes(self.data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
            self.data = swapaxes(self.data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
        elif perm == [1, 2, 0]:
            self.data = swapaxes(self.data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
            self.data = swapaxes(self.data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
        elif perm == [0, 1, 2]:
            # do nothing
            pass
        else:
            print 'Error: wrong orientation'
        # update dim
        # http://math.stackexchange.com/questions/122916/what-is-the-inverse-cycle-of-permutation
        dim_temp = list(self.dim)
        dim_temp[0] = self.dim[[i for i, x in enumerate(perm) if x == 0][0]]  # nx
        dim_temp[1] = self.dim[[i for i, x in enumerate(perm) if x == 1][0]]  # ny
        dim_temp[2] = self.dim[[i for i, x in enumerate(perm) if x == 2][0]]  # nz
        dim_temp[4] = self.dim[[i for i, x in enumerate(perm) if x == 0][0]+4]  # px
        dim_temp[5] = self.dim[[i for i, x in enumerate(perm) if x == 1][0]+4]  # py
        dim_temp[6] = self.dim[[i for i, x in enumerate(perm) if x == 2][0]+4]  # pz
        self.dim = tuple(dim_temp)
        # update orientation
        self.orientation = orientation
        return raw_orientation

    def show(self):
        from matplotlib.pyplot import imshow, show
        imgplot = imshow(self.data)
        imgplot.set_cmap('gray')
        imgplot.set_interpolation('nearest')
        show()

    def transfo_pix2phys(self, coordi=None):
        """


        This function returns the physical coordinates of all points of 'coordi'. 'coordi' is a list of list of size
        (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical
        coordinates of the points in the space of the image.

        Example:
        img = Image('file.nii.gz')
        coordi_pix = [[1,1,1],[2,2,2],[4,4,4]]   # for points: (1,1,1), (2,2,2) and (4,4,4)
        coordi_phys = img.transfo_pix2phys(coordi=coordi_pix)

        :return:
        """
        from numpy import zeros, array, transpose, dot, asarray

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3,0:3]
        coord_origin = array([[m_p2f[0, 3]],[m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi != None:
            coordi_pix = transpose(asarray(coordi))
            coordi_phys = transpose(coord_origin + dot(m_p2f_transfo, coordi_pix))
            coordi_phys_list = coordi_phys.tolist()

            return coordi_phys_list

    def transfo_phys2pix(self, coordi=None):
        """
        This function returns the pixels coordinates of all points of 'coordi'
        'coordi' is a list of list of size (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical coordinates of the points in the space of the image.


        :return:
        """
        from numpy import array, transpose, dot, asarray
        from numpy.linalg import inv

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3,0:3]
        m_f2p_transfo = inv(m_p2f_transfo)

        coord_origin = array([[m_p2f[0, 3]],[m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi != None:
            coordi_phys = transpose(asarray(coordi))
            coordi_pix =  transpose(dot(m_f2p_transfo, (coordi_phys-coord_origin)))
            coordi_pix_tmp = coordi_pix.tolist()
            coordi_pix_list = [[int(round(coordi_pix_tmp[j][i])) for i in range(len(coordi_pix_tmp[j]))] for j in range(len(coordi_pix_tmp))]

            return coordi_pix_list

    def transfo_phys2continuouspix(self, coordi=None, data_phys=None):
        """
        This function returns the pixels coordinates of all points of data_pix in the space of the image. The output is a matrix of size: size(data_phys) but containing a 3D vector.
        This vector is the pixel position of the point in the space of the image.
        data_phys must be an array of 3 dimensions for which each point contains a vector (physical position of the point).

        If coordi is different from none:
        coordi is a list of list of size (nb_points * 3) containing the pixel coordinate of points. The function will return a list with the physical coordinates of the points in the space of the image.


        :return:
        """
        from numpy import array, transpose, dot, asarray
        from numpy.linalg import inv
        from copy import copy

        m_p2f = self.hdr.get_sform()
        m_p2f_transfo = m_p2f[0:3, 0:3]
        m_f2p_transfo = inv(m_p2f_transfo)
        # e = dot(m_p2f_transfo, m_f2p_transfo)

        coord_origin = array([[m_p2f[0, 3]], [m_p2f[1, 3]], [m_p2f[2, 3]]])

        if coordi != None:
            coordi_phys = transpose(asarray(coordi))
            coordi_pix = transpose(dot(m_f2p_transfo, (coordi_phys - coord_origin)))
            coordi_pix_tmp = coordi_pix.tolist()
            coordi_pix_list = [[coordi_pix_tmp[j][i] for i in range(len(coordi_pix_tmp[j]))] for j in
                               range(len(coordi_pix_tmp))]

            return coordi_pix_list


def find_zmin_zmax(fname):
    import sct_utils as sct
    # crop image
    status, output = sct.run('sct_crop_image -i '+fname+' -dim 2 -bmax -o tmp.nii')
    # parse output
    zmin, zmax = output[output.find('Dimension 2: ')+13:].split('\n')[0].split(' ')
    return int(zmin), int(zmax)


def get_dimension(im_file, verbose=1):
    """
    Get dimension from nibabel object. Manages 2D, 3D or 4D images.
    :return: nx, ny, nz, nt, px, py, pz, pt
    """
    import nibabel.nifti1
    import sct_utils as sct
    # initialization
    nx, ny, nz, nt, px, py, pz, pt = 1, 1, 1, 1, 1, 1, 1, 1
    if type(im_file) is nibabel.nifti1.Nifti1Image:
        header = im_file.header
    elif type(im_file) is Image:
        header = im_file.hdr
    else:
        header = None
        sct.printv('WARNING: the provided image file isn\'t a nibabel.nifti1.Nifti1Image instance nor a msct_image.Image instance', verbose, 'warning')
    nb_dims = len(header.get_data_shape())
    if nb_dims == 2:
        nx, ny = header.get_data_shape()
        px, py = header.get_zooms()
    if nb_dims == 3:
        nx, ny, nz = header.get_data_shape()
        px, py, pz = header.get_zooms()
    if nb_dims == 4:
        nx, ny, nz, nt = header.get_data_shape()
        px, py, pz, pt = header.get_zooms()

    return nx, ny, nz, nt, px, py, pz, pt


def change_data_orientation(data, old_orientation='RPI', orientation="RPI"):
    """
    This function changes the orientation of a data matrix from a give orientation to another.
    This function assumes that the user already knows the orientation of the data
    :param data: data of the image
    :param old_orientation: Current orientation of the data
    :param orientation: Desired orientation for the data
    :return: Data matrix representing the
    """
    opposite_character = {'L': 'R', 'R': 'L', 'A': 'P', 'P': 'A', 'I': 'S', 'S': 'I'}

    # change the orientation of the image
    perm = [0, 1, 2]
    inversion = [1, 1, 1]
    for i, character in enumerate(old_orientation):
        try:
            perm[i] = orientation.index(character)
        except ValueError:
            perm[i] = orientation.index(opposite_character[character])
            inversion[i] = -1

    # axes inversion
    data = data[::inversion[0], ::inversion[1], ::inversion[2]]

    # axes manipulations
    from numpy import swapaxes

    if perm == [1, 0, 2]:
        data = swapaxes(data, 0, 1)
    elif perm == [2, 1, 0]:
        data = swapaxes(data, 0, 2)
    elif perm == [0, 2, 1]:
        data = swapaxes(data, 1, 2)
    elif perm == [2, 1, 0]:
        data = swapaxes(data, 0, 2)
    elif perm == [2, 0, 1]:
        data = swapaxes(data, 0, 2)  # transform [2, 0, 1] to [1, 0, 2]
        data = swapaxes(data, 0, 1)  # transform [1, 0, 2] to [0, 1, 2]
    elif perm == [1, 2, 0]:
        data = swapaxes(data, 0, 2)  # transform [1, 2, 0] to [0, 2, 1]
        data = swapaxes(data, 1, 2)  # transform [0, 2, 1] to [0, 1, 2]
    elif perm == [0, 1, 2]:
        # do nothing
        pass
    else:
        print 'Error: wrong orientation'

    return data

# =======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    from msct_parser import Parser
    from sct_utils import add_suffix
    import sys

    parser = Parser(__file__)
    parser.usage.set_description('Image processing functions')
    parser.add_option(name="-i",
                      type_value="file",
                      description="Image input file.",
                      mandatory=True,
                      example='im.nii.gz')
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Image output name.",
                      mandatory=False,
                      example='im_out.nii.gz')


    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    image.changeType('minimize')
    name_out = ''
    if "-o" in arguments:
        name_out = arguments["-o"]
