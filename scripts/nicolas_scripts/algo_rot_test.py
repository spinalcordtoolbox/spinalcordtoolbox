



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
        for filename in fnmatch.filter(filenames, "*.nii"):  # if file with nii extension (.nii or .nii.gz) found
            if "seg" in filename:
                continue  # do not consider it if it's a segmentation
            if filename.split(".nii")[0] + "_seg.nii" in filenames:  # find potential seg associated with the file
                file_seg_input = os.path.join(root, filename.split(".nii")[0] + "_seg.nii")
            else:
                file_seg_input = None

            if test_str == "test_list_folder":
                test_list_folder(file_input=os.path.join(root, filename), path_output=path_output)
            elif test_str == "test_2D_hogancest":
                test_2D_hogancest(file_input=os.path.join(root, filename),
                                  path_output=path_output, file_seg_input=file_seg_input)
            else:
                raise Exception("no such test as " + test_str + " exists")


def test_list_folder(file_input, path_output):

    sct.printv("input " + file_input + "\n ouput " + path_output)

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
    else:
        image_seg_data = load_image(file_input=file_seg_input, dimension=2)
        centermass = np.round(scipy.ndimage.measurements.center_of_mass(image_seg_data))

        (nx, ny) = image_data.shape
        xx, yy = np.mgrid[:nx, :ny]
        mask = np.zeros((nx, ny))
        mask = np.exp(
            -(((xx - centermass[0]) ** 2) / (2 * (sigma ** 2)) + ((yy - centermass[1]) ** 2) / (2 * (sigma ** 2))))

    if np.isnan(image_data).any() or np.isnan(centermass).any():  # if nan present in image_data or in centermass
        sct.printv("image corrupted or segmentation = 0 everywhere in this slice for image : " + file_input)
        return  # exit function

    # Finding axes of symmetry
    hog_ancest, weighting_map = hog_ancestor(image_data, nb_bin=nb_bin, seg_probability_map=mask, return_image=True)
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
    plt.subplot(235)
    plt.imshow(mask)
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
    image_wline = image_data
    for i_angle in range(0, len(angles)):
        image_wline = generate_2Dimage_line(image_wline, centermass[0], centermass[1], angles[i_angle])

    # Saving image with axes drawn on it
    save_image(image_wline, "sym_" + (file_input.split("/")[-1]).split(".nii")[0] + ".nii",
               file_input, ofolder=path_output)  # the character manipulation permits to have the name of the file
    #  which is at the end of the path ( / ) and just before .nii
    cwd = os.getcwd()
    os.chdir(path_output)
    plt.savefig("results_" + (file_input.split("/")[-1]).split(".nii")[0] + ".png")
    plt.close()


def load_image(file_input, dimension):
    """ Users asks what dimension he wants in output
        """
    # This function's purpose is for testing, it is not clean

    if dimension == 3:
        image_data = np.array(Image(file_input).data) # just retrieve the data
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
