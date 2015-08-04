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

class Image(object):
    """

    """
    def __init__(self, param=None, hdr=None, orientation=None, absolutepath="", verbose=1, split=False):
        from numpy import zeros, ndarray, generic
        from sct_utils import extract_fname

        # initialization of all parameters
        self.data = None
        self.hdr = None
        self.orientation = None
        self.absolutepath = ""
        self.path = ""
        self.file_name = ""
        self.ext = ""
        self.dim = None

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
            self.dim = self.data.shape
            self.hdr = hdr
            self.orientation = orientation
            self.absolutepath = absolutepath
            self.path, self.file_name, self.ext = extract_fname(absolutepath)
        else:
            raise TypeError('Image constructor takes at least one argument.')

        """
        if split:
            self.data = self.split_data()
        """

    def __deepcopy__(self, memo):
        from copy import deepcopy
        return type(self)(deepcopy(self.data,memo),deepcopy(self.hdr,memo),deepcopy(self.orientation,memo),deepcopy(self.absolutepath,memo))

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
        from sct_utils import check_file_exist, printv, extract_fname, get_dimension
        from sct_orientation import get_orientation

        check_file_exist(path, verbose=verbose)
        try:
            im_file = load(path)
        except spatialimages.ImageFileError:
            printv('Error: make sure ' + path + ' is an image.', 1, 'error')
        self.orientation = get_orientation(path)
        self.data = im_file.get_data()
        self.hdr = im_file.get_header()
        self.absolutepath = path
        self.path, self.file_name, self.ext = extract_fname(path)
        nx, ny, nz, nt, px, py, pz, pt = get_dimension(path)
        self.dim = [nx, ny, nz]

    def setFileName(self, filename):
        from sct_utils import extract_fname
        self.absolutepath = filename
        self.path, self.file_name, self.ext = extract_fname(filename)

    def changeType(self, type=''):
        from numpy import uint8, uint16, uint32, uint64, int8, int16, int32, int64, float32, float64

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
        :param type:    if not set, the image is saved in standard type
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

        if type != '':
            self.changeType(type)

        self.hdr.set_data_shape(self.data.shape)
        img = Nifti1Image(self.data, None, self.hdr)
        #printv('saving ' + self.path + self.file_name + self.ext + '\n', self.verbose)
        save(img, self.path + self.file_name + self.ext)

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

        X, Y, Z = (self.data > 0.0).nonzero()
        if coordValue:
            from msct_types import CoordinateValue
            list_coordinates = [CoordinateValue([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]
        else:
            from msct_types import Coordinate
            list_coordinates = [Coordinate([X[i], Y[i], Z[i], self.data[X[i], Y[i], Z[i]]]) for i in range(0, len(X))]

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
    # This method is called in sct_crop_over_mask script
    def crop_from_square_mask(self, mask):
        from numpy import asarray

        array = self.data
        data_mask = mask.data
        print 'ORIGINAL SHAPE: ', array.shape, '   ==   ', data_mask.shape
        array = asarray(array)
        data_mask = asarray(data_mask)
        new_data = []
        buffer = []
        buffer_mask = []
        s = 0
        r = 0
        ok = 0
        for slice in data_mask:
            # print 'SLICE ', s, slice
            for row in slice:
                if sum(row) > 0:
                    buffer_mask.append(row)
                    buffer.append(array[s][r])
                    #print 'OK1', ok
                    ok += 1
                r += 1
            new_slice_mask = asarray(buffer_mask).T
            new_slice = asarray(buffer).T
            r = 0
            buffer = []
            for row in new_slice_mask:
                if sum(row) != 0:
                    buffer.append(new_slice[r])
                r += 1
            #print buffer
            new_slice = asarray(buffer).T
            r = 0
            buffer_mask = []
            buffer = []
            new_data.append(new_slice)
            s += 1
        new_data = asarray(new_data)
        # print data_mask
        print 'SHAPE ', new_data.shape
        self.data = new_data

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
        elif perm == [2, 1, 0]:
            self.data = swapaxes(self.data, 0, 2)
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

        self.orientation = orientation

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


def pad_image(fname_in, file_out, padding):
    import sct_utils as sct
    sct.run('isct_c3d '+fname_in+' -pad 0x0x'+str(padding)+'vox 0x0x'+str(padding)+'vox 0 -o '+file_out, 1)
    return


def find_zmin_zmax(fname):
    import sct_utils as sct
    # crop image
    status, output = sct.run('sct_crop_image -i '+fname+' -dim 2 -bmax -o tmp.nii')
    # parse output
    zmin, zmax = output[output.find('Dimension 2: ')+13:].split('\n')[0].split(' ')
    return int(zmin), int(zmax)


# =======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    from msct_parser import Parser
    import sys

    parser = Parser(__file__)
    parser.usage.set_description('Image')
    parser.add_option("-i", "file", "file", True)
    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    image.changeType('minimize')