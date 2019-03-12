



import os
import numpy as np
from spinalcordtoolbox.image import Image
import matplotlib.pyplot as plt
from msct_parser import Parser
import sct_utils as sct
import sys, os, shutil
from functions_sym_rot import *
import fnmatch
import scipy
from msct_register import compute_pca, angle_between

def get_parser():

    parser = Parser(__file__)
    parser.usage.set_description('Blablablabla')
    parser.add_option(name="-i",
                      type_value="folder",
                      description="Input folder with data to test",
                      mandatory=True,
                      example="/home/data")

    parser.add_option(name="-test",  # TODO find better name
                      type_value="str",
                      description="put name of the test you want to run",
                      mandatory=True,
                      example="src_seg.nii.gz")
    parser.add_option(name="-o",
                      type_value="folder",
                      description="output folder for test results",
                      mandatory=False,
                      example="path/to/output/folder")

    return parser



def main(args=None):

    if args is None:
        args = sys.argv[1:]

    parser = get_parser()
    arguments = parser.parse(args)
    input_folder = arguments['-i']
    test_str = arguments['-test']
    if '-o' in arguments:
        path_output = arguments['-o']
    else:
        path_output = os.getcwd()

    for root, dirnames, filenames in os.walk(input_folder):  # searching the given directory
        for filename in fnmatch.filter(filenames, "*.nii*"):  # if file with nii extension (.nii or .nii.gz) found
            if "seg" in filename:
                continue  # do not consider it if it's a segmentation
            if "dwi" in filename:
                continue  # do not consider it if it's DWI
            if filename.split(".nii")[0] + "_seg.nii" + filename.split(".nii")[1] in filenames:  # find potential seg associated with the file
                file_seg_input = os.path.join(root, filename.split(".nii")[0] + "_seg.nii" + filename.split(".nii")[1])
            else:
                file_seg_input = None

            if test_str == "test_list_folder":
                test_list_folder(file_input=os.path.join(root, filename), path_output=path_output)
            elif test_str == "test_2D_hogancest":
                test_2D_hogancest(file_input=os.path.join(root, filename),
                                  path_output=path_output, file_seg_input=file_seg_input)
            elif test_str == "test_3D_hogancest":
                if file_seg_input is None:
                    sct.printv("no segmentation for file : " + filename)
                    continue
                test_3D_hogancest(file_input=os.path.join(root, filename),
                                  path_output=path_output, file_seg_input=file_seg_input)
            elif test_str == "test_3D_PCA":
                if file_seg_input is None:
                    sct.printv("no segmentation for file : " + filename)
                    continue
                test_3D_PCA(file_input=os.path.join(root, filename),
                            path_output=path_output, file_seg_input=file_seg_input)
            elif test_str == "compare_PCA_Hogancest":
                if file_seg_input is None:
                    sct.printv("no segmentation for file : " + filename)
                    continue
                compare_PCA_Hogancest(file_input=os.path.join(root, filename),
                            path_output=path_output, file_seg_input=file_seg_input)
            else:
                raise Exception("no such test as " + test_str + " exists")


def test_list_folder(file_input, path_output):

    sct.printv("input " + file_input + "\n ouput " + path_output)

def compare_PCA_Hogancest(file_input, file_seg_input, path_output):

    # parameters :
    sigma = 10
    nb_bin = 360
    kmedian_size = 3
    angle_range = 30
    # TODO implemente kernel gradient size parameter

    image = Image(file_input)
    image_data = image.data
    seg = Image(file_seg_input)
    seg_data = seg.data
    if image_data.shape != seg_data.shape:
        raise Exception("error, data and seg have not the same dimension")
    if len(image_data.shape) > 3:
        raise Exception("error, data is not 3D")
    nx, ny, nz, _, px, py, pz, _ = image.dim
    mask_rot = np.zeros((nx, ny, nz))
    weighting_map = np.zeros((nx, ny, nz))
    angles_slices = np.zeros((2, nz))

    sigmax = sigma/px
    sigmay = sigma/py

    for zslice in range(0, nz):

        slice_image = image_data[:, :, zslice]
        slice_seg = seg_data[:, :, zslice]
        if not np.any(slice_seg):  # if no segmentation on that slice
            angles_slices[:, zslice] = -120
            continue

        # PCA :

        _, pca, centermass = compute_pca(slice_seg)
        eigenv = pca.components_.T[0][0], pca.components_.T[1][0]
        angle_found = 180 / pi * angle_between((0, 1), eigenv)
        if -180 <= angle_found < -90:  # translate angles that are not between -90 abd 90
            angle_found = 180 + angle_found
        elif 90 < angle_found <= 180:
            angle_found = 180 - angle_found

        if -90 <= angle_found <= 0:  # just a quick switch of origin for visualisation purpose
            angles_slices[0, zslice] = angle_found + 90
        else:  # 0 < angle_found <= 90
            angles_slices[0, zslice] = angle_found - 90

        mask_rot[:, :, zslice] = generate_2Dimage_line(mask_rot[:, :, zslice], centermass[0], centermass[1], angle_found, value=1)

        # Hogancest :

        centermass = np.round(scipy.ndimage.measurements.center_of_mass(slice_seg))
        xx, yy = np.mgrid[:nx, :ny]
        seg_weighted_mask = np.exp(
            -(((xx - centermass[0]) ** 2) / (2 * (sigmax ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigmay ** 2))))

        hog_ancest, slice_weighting_map = hog_ancestor(slice_image, nb_bin=nb_bin, seg_weighted_mask=seg_weighted_mask,
                                                 return_image=True)
        weighting_map[:, :, zslice] = slice_weighting_map
        hog_ancest_smooth = circular_filter_1d(hog_ancest, kmedian_size,
                                               filter='median')  # fft than square than ifft to calculate convolution
        hog_fft2 = np.fft.rfft(hog_ancest_smooth) ** 2
        hog_conv = np.real(np.fft.irfft(hog_fft2))

        hog_conv_reordered = np.zeros(nb_bin)
        hog_conv_reordered[0:180] = hog_conv[180:360]
        hog_conv_reordered[180:360] = hog_conv[0:180]
        hog_conv_restrained = hog_conv_reordered[nb_bin/2-np.true_divide(angle_range, 180)*nb_bin:nb_bin/2+np.true_divide(angle_range, 180)*nb_bin]

        argmaxs = argrelextrema(hog_conv_restrained, np.greater, mode='wrap', order=kmedian_size)[0]  # get local maxima
        argmaxs_sorted = np.asarray([tutut for _, tutut in
                                     sorted(zip(hog_conv_restrained[argmaxs], argmaxs),
                                            reverse=True)])  # sort maxima based on value
        argmaxs_sorted = (argmaxs_sorted - nb_bin / 2) * np.true_divide(180, nb_bin*angle_range)  # angles are computed from -angle_range to angle_range
        if len(argmaxs_sorted) == 0:  # no angle found
            angles_slices[1, zslice] = -140

        else:
            angle_found = argmaxs_sorted[0]
            if -90 < angle_found <= 0:  # just a quick switch of origin because np array are not oriented the same as images
                angle_draw = angle_found + 90
            else:  # 0 < angle_found <= 90
                angle_draw = angle_found - 90
            mask_rot[:, :, zslice] = generate_2Dimage_line(mask_rot[:, :, zslice], centermass[0], centermass[1], angle_draw, value=2)
            angles_slices[1, zslice] = angle_found
        1+1

    path_output = path_output + "/" + (file_input.split("/")[-1]).split(".nii")[0]
    if not os.path.exists(path_output):
        os.makedirs(path_output)

    save_image(mask_rot, (file_input.split("/")[-1]).split(".nii")[0] + "_RotMask" + ".nii.gz",
               file_input, ofolder=path_output)
    save_image(weighting_map, (file_input.split("/")[-1]).split(".nii")[0] + "_weightmap" + ".nii.gz",
               file_input, ofolder=path_output)
    sct.copy(file_input, path_output + "/" + file_input.split("/")[-1], verbose=0)  # copy original file to help visualization
    sct.copy(file_seg_input, path_output + "/" + file_seg_input.split("/")[-1], verbose=0)

    plt.figure(figsize=(8, 8))
    plt.title("Results for file : " + file_input.split("/")[-1])
    plt.plot(range(0, nz), angles_slices[0, :], "bo")
    plt.plot(range(0, nz), angles_slices[1, :], "ro")
    plt.legend(("PCA", "Hogancest"))
    plt.xlabel("no slice")
    plt.ylabel("angle found in degrees")

    os.chdir(path_output)
    plt.savefig("angles_for_V2_" + (file_input.split("/")[-1]).split(".nii")[0] + ".png")
    plt.close()

    sct.display_viewer_syntax([file_input, file_seg_input, path_output + "/" + (file_input.split("/")[-1]).split(".nii")[0] + "_RotMask" + ".nii.gz"],
                              colormaps=['gray', 'red', 'random'], opacities=['', '0.7', '1'])


def test_3D_hogancest(file_input, file_seg_input, path_output):

    # parameters :
    sigma = 10
    nb_bin = 360
    kmedian_size = 3
    angle_range = 30
    # TODO implemente kernel gradient size parameter

    image = Image(file_input)
    image_data = image.data
    seg = Image(file_seg_input)
    seg_data = seg.data
    if image_data.shape != seg_data.shape:
        raise Exception("error, data and seg have not the same dimension")
    if len(image_data.shape) > 3:
        raise Exception("error, data is not 3D")
    nx, ny, nz = image_data.shape
    imagerot_data = np.zeros((nx, ny, nz))
    angles_slices = np.zeros(nz)

    for zslice in range(0, nz):

        slice_image = image_data[:, :, zslice]
        slice_seg = seg_data[:, :, zslice]
        if not np.any(slice_seg):  # if no segmentation on that slice
            angles_slices[zslice] = -200
            imagerot_data[:, :, zslice] = slice_image
            continue

        centermass = np.round(scipy.ndimage.measurements.center_of_mass(slice_seg))
        xx, yy = np.mgrid[:nx, :ny]
        seg_weighted_mask = np.exp(
            -(((xx - centermass[0]) ** 2) / (2 * (sigma ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigma ** 2))))

        hog_ancest, weighting_map = hog_ancestor(slice_image, nb_bin=nb_bin, seg_weighted_mask=seg_weighted_mask,
                                                 return_image=True)

        hog_ancest_smooth = circular_filter_1d(hog_ancest, kmedian_size,
                                               filter='median')  # fft than square than ifft to calculate convolution
        hog_fft2 = np.fft.rfft(hog_ancest_smooth) ** 2
        hog_conv = np.real(np.fft.irfft(hog_fft2))
        argmaxs = argrelextrema(hog_conv, np.greater, mode='wrap', order=kmedian_size)[0]  # get local maxima

        argmaxs_sorted = np.asarray([tutut for _, tutut in
                                     sorted(zip(hog_conv[argmaxs], argmaxs),
                                            reverse=True)])  # sort maxima based on value
        argmaxs_sorted = (argmaxs_sorted - nb_bin / 2) * 180 / nb_bin  # angles are computed from -90 to 90
        argmaxs_sorted = -1 * argmaxs_sorted  # not sure why but angles are are positive clockwise (inverse convention)
        if len(argmaxs_sorted) == 0:
            angle_found = None
            imagerot_data[:, :, zslice] = slice_image
            angles_slices[zslice] = -200  # TODO : difference between no angle found and no seg

        else:
            angle_found = argmaxs_sorted[0]
            imagerot_data[:, :, zslice] = generate_2Dimage_line(slice_image, centermass[0], centermass[1], angle_found)
            if -90 < angle_found <= 0:  # just a quick switch of origin for visualisation purpose
                angles_slices[zslice] = angle_found + 90
            else:  # 0 < angle_found <= 90
                angles_slices[zslice] = angle_found - 90

    save_image(imagerot_data, "symHogancest_" + (file_input.split("/")[-1]).split(".nii")[0] + ".nii",
               file_input, ofolder=path_output)

    plt.figure(figsize=(8, 8))
    plt.title("Results for file : " + file_input.split("/")[-1])
    plt.scatter(range(0, nz), angles_slices, c=np.where((angles_slices < 30) * (angles_slices > -30), 0, 1))
    plt.xlabel("no slice")
    plt.ylabel("angle found in degrees")

    os.chdir(path_output)
    plt.savefig("angles_for_" + (file_input.split("/")[-1]).split(".nii")[0] + ".png")
    plt.close()

def test_3D_PCA(file_input, file_seg_input, path_output):

    image = Image(file_input)
    image_data = image.data
    seg = Image(file_seg_input)
    seg_data = seg.data
    nx, ny, nz = seg_data.shape
    imagerot_data = np.zeros((nx, ny, nz))
    angles_slices = np.zeros(nz)

    for zslice in range(0, nz):

        slice_seg = seg_data[:, :, zslice]
        slice_image = image_data[:, :, zslice]
        if not np.any(slice_seg):  # if no segmentation on that slice
            angles_slices[zslice] = -200
            imagerot_data[:, :, zslice] = slice_image
            continue

        _, pca, centermass = compute_pca(slice_seg)
        eigenv = pca.components_.T[0][0], pca.components_.T[1][0]
        angle_found = 180/pi * angle_between((0, 1), eigenv)
        imagerot_data[:, :, zslice] = generate_2Dimage_line(slice_image, centermass[0], centermass[1], angle_found)
        angles_slices[zslice] = angle_found

    save_image(imagerot_data, "symPCA_" + (file_input.split("/")[-1]).split(".nii")[0] + ".nii",
               file_input, ofolder=path_output)

    plt.figure(figsize=(8, 8))
    plt.title("Results for file : " + file_input.split("/")[-1])
    plt.scatter(range(0, nz), angles_slices, c=np.where((angles_slices < 30) * (angles_slices > -30), 0, 1))
    plt.xlabel("no slice")
    plt.ylabel("angle found in degrees")

    os.chdir(path_output)
    plt.savefig("angles_for_" + (file_input.split("/")[-1]).split(".nii")[0] + ".png")
    plt.close()


def test_2D_hogancest(file_input, path_output, file_seg_input=None):

    # Params
    nb_axes = 1  # put -1 for all axes found
    kmedian_size = 3
    nb_bin = 360
    sigma = 10  # TODO : create 2 sigmas (x and y) and make it dependent of size of segmentation

    # Loading image
    image_data = load_image(file_input=file_input, dimension=2)

    # Center of mass (to draw axes later or to create mask)
    if file_seg_input is None:
        # centermass = image[0].mean(1).round().astype(int)  # will act weird if image is non binary
        centermass = [int(round(image_data.shape[0] / 2)), int(round(image_data.shape[1] / 2))]  # center of image
        seg_weighted_mask = None
        (nx, ny) = image_data.shape
    else:
        image_seg_data = load_image(file_input=file_seg_input, dimension=2)
        centermass = np.round(scipy.ndimage.measurements.center_of_mass(image_seg_data))

        (nx, ny) = image_data.shape  # TODO : make this a function, make_mask_seg or smth
        xx, yy = np.mgrid[:nx, :ny]
        seg_weighted_mask = np.exp(
            -(((xx - centermass[0]) ** 2) / (2 * (sigma ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigma ** 2))))

    if np.isnan(image_data).any() or np.isnan(centermass).any():  # if nan present in image_data or in centermass
        sct.printv("image corrupted or segmentation = 0 everywhere in this slice for image : " + file_input)
        return  # exit function

    # Finding axes of symmetry
    hog_ancest, weighting_map = hog_ancestor(image_data, nb_bin=nb_bin,
                                             seg_weighted_mask=seg_weighted_mask, return_image=True)
    # smooth it with median filter
    hog_ancest_smooth = circular_filter_1d(hog_ancest, kmedian_size,
                                           filter='median')  # fft than square than ifft to calculate convolution
    hog_fft2 = np.fft.rfft(hog_ancest_smooth) ** 2
    hog_conv = np.real(np.fft.irfft(hog_fft2))
        # TODO FFT CHECK SAMPLING
        # hog_conv = np.convolve(hog_ancest_smooth, hog_ancest_smooth, mode='same')
    # search for maximum to find angle of rotation
    argmaxs = argrelextrema(hog_conv, np.greater, mode='wrap', order=kmedian_size)[0]  # get local maxima
    argmaxs_sorted = np.asarray([tutut for _, tutut in
                      sorted(zip(hog_conv[argmaxs], argmaxs), reverse=True)])  # sort maxima based on value
    # argmaxs_sorted_nodouble = argmaxs_sorted[np.where(argmaxs_sorted >= 0)]
    argmaxs_sorted = (argmaxs_sorted - nb_bin/2) * 180/nb_bin  # angles are computed from -90 to 90
    argmaxs_sorted = -1 * argmaxs_sorted  # not sure why but angles are are positive clockwise (inverse convention)

    plt.figure(figsize=(20, 10))
    plt.suptitle("angle found : " + str(argmaxs_sorted[0]))
    plt.subplot(231)
    plt.plot(np.arange(-180, 180, 1), hog_ancest)
    plt.title("hog_ancest")
    plt.subplot(232)
    plt.plot(np.arange(-180, 180, 1), hog_ancest_smooth)
    plt.title("hog_ancest_smooth")
    plt.subplot(233)
    plt.plot(np.arange(-90, 90, 0.5), hog_conv)
    plt.title("hog_conv")
    plt.subplot(234)
    plt.imshow(255 - image_data, cmap="Greys")
    plt.title((file_input.split("/")[-1]).split(".nii")[0])
    if seg_weighted_mask is not None:
        plt.subplot(235)
        plt.imshow(seg_weighted_mask)
        plt.title("segmentation weighted map")
        plt.colorbar()
    plt.subplot(236)
    plt.imshow(weighting_map)
    plt.colorbar()
    plt.title("final weighting map")

    if nb_axes == -1:
        angles = argmaxs_sorted
    elif nb_axes > len(argmaxs_sorted):
        sct.printv("For file " + file_input + str(nb_axes) +
                   " axes of symmetry were asked for, only found " + str(len(argmaxs_sorted)))
        angles = argmaxs_sorted
    else:
        angles = argmaxs_sorted[0:nb_axes]


    # Draw axes on image
    # image_wline = image_data
    # for i_angle in range(0, len(angles)):
    #     image_wline = generate_2Dimage_line(image_wline, centermass[0], centermass[1], angles[i_angle])


    # # Saving image with axes drawn on it
    # save_image(image_wline, "sym_" + (file_input.split("/")[-1]).split(".nii")[0] + ".nii",
    #            file_input, ofolder=path_output)  # the character manipulation permits to have the name of the file
    # #  which is at the end of the path ( / ) and just before .nii

    for i_angle in range(0, len(angles)):
        mask_rot = generate_2Dimage_line(np.zeros((nx, ny)), centermass[0], centermass[1], angles[i_angle], value=i_angle+1)

    save_image(mask_rot,  (file_input.split("/")[-1]).split(".nii")[0] + "_mask_sym.nii",
                           file_input, ofolder=path_output)
    save_image(image_data, file_input.split("/")[-1], file_input, ofolder=path_output) # quick copy of image in output folder

    os.chdir(path_output)
    plt.savefig("nice_fig_" + (file_input.split("/")[-1]).split(".nii")[0] + ".png")
    plt.close()


def load_image(file_input, dimension):
    """ Users asks what dimension he wants in output
        """
    # This function's purpose is for testing, it is not clean

    if dimension == 3:
        image_data = np.array(Image(file_input).data)  # just retrieve the data
        if len(image_data.shape) != 3:
            raise Exception("Dimension said to be 3 but is " + str(len(image_data.shape)))

    elif dimension == 2:
        image_data = np.array(Image(file_input).data) # retrieve data
        if len(image_data.shape) == 3:
            image_data = np.mean(np.array(Image(file_input).data), axis=2) #but mean because 3r axe might be rgb

        if len(image_data.shape) != 2:
            raise Exception("Dimension said to be 2 but is " + str(len(image_data.shape)))

    else:
        raise Exception("Dimension input must be 2 or 3, not " + str(dimension))

    return image_data

def save_image(data, fname, fname_like, ofolder=None):
    """ This functions creates a nifti image with data provided and the same header as the file provided
    inputs :
        - data : ND numpy array of data that we want to save as nifti
        - fname : name wanted for the data, an str
        - fname_like : name of the file that the header will be copied from to form the image, an str
    outputs :
        - the output image is saved under the name fname, contains the data data and the header of the fname_like file
        """

    img_like = load(fname_like)
    header = img_like.header.copy()
    img = Nifti1Image(data, None, header=header)
    if ofolder is not None:
        cwd = os.getcwd()
        os.chdir(ofolder)
    save(img, fname)
    if ofolder is not None:
        os.chdir(cwd)


if __name__ == "__main__":
    sct.init_sct()
    # call main function
    main()
