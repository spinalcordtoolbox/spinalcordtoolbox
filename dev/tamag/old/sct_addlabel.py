#!/usr/bin/env python

import numpy as np
import commands, sys


# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))

from msct_image import Image
from msct_types import Coordinate
from msct_parser import Parser
from sct_label_utils import ProcessLabels

from sct_get_centerline_from_labels import ComputeZMinMax


#definition
#take file of entry find Zmax and Zmin of the propseg segmentation (segmented file)
# use Zmax to add one label to the brainstem labels (brainstem file) at the Zmax-10 slice

#segmented_file                  file to extract zmax
#brainstem file                  file with labels only
#label_depth_compared_to_zmax    depth wanted to add the label
#label_value                     label value (set to 1 by default)

def add_label(brainstem_file, segmented_file, output_file_name, label_depth_compared_to_zmax=10 , label_value=1):
    #Calculating zmx of the segmented file  (data_RPI_seg.nii.gz)
    image_seg = Image(segmented_file)
    z_test = ComputeZMinMax(image_seg)
    zmax = z_test.Zmax
    print( "Zmax: ",zmax)

    #Test on the number of labels
    brainstem_image = Image(brainstem_file)
    print("nb_label_before=", np.sum(brainstem_image.data))

    #Center of mass
    X, Y = np.nonzero((z_test.image.data[:,:,zmax-label_depth_compared_to_zmax] > 0))
    x_bar = 0
    y_bar = 0
    for i in range(X.shape[0]):
        x_bar = x_bar+X[i]
        y_bar = y_bar+Y[i]
    x_bar = int(round(x_bar/X.shape[0]))
    y_bar = int(round(y_bar/X.shape[0]))

    #Placement du nouveau label aux coordonnees x_bar, y_bar et zmax-label_depth_compared_to_zmax
    coordi = Coordinate([x_bar, y_bar, zmax-label_depth_compared_to_zmax, label_value])
    object_for_process = ProcessLabels(brainstem_file, coordinates=[coordi])
    #print("object_for_process.coordinates=", object_for_process.coordinates.x, object_for_process.coordinates.y, object_for_process.coordinates.z)
    file_with_new_label=object_for_process.create_label()

    #Define output file
    im_output = object_for_process.image_input.copy()
    im_output.data *= 0
    brainstem_image=Image(brainstem_file)
    im_output.data = brainstem_image.data + file_with_new_label.data

    #Test the number of labels
    print("nb_label_after=", np.sum(im_output.data))

    #Save output file
    im_output.setFileName(output_file_name)
    im_output.save('minimize')



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters


    # Initialize the parser
    parser = Parser(__file__)
    parser.usage.set_description('Utility function for labels.')
    parser.add_option(name="-i",
                      type_value="file",
                      description="image to add labels on. Must be 3D.",
                      mandatory=True,
                      example="t2_labels.nii.gz")
    parser.add_option(name="-c",
                      type_value="file",
                      description="file presenting the segmentation. Must be 3D.",
                      mandatory=True)
    parser.add_option(name="-o",
                      type_value="file_output",
                      description="Name of the output niftii image.",
                      mandatory=False,
                      example="t2_labels_cross.nii.gz",
                      default_value="labels.nii.gz")
    parser.add_option(name="-d",
                      type_value="int",
                      description="depth wanted for the label compared to zmax",
                      mandatory=False,
                      default_value="10")
    parser.add_option(name="-v",
                      type_value="int",
                      description="wanted value for the label to add",
                      mandatory=False,
                      default_value="1")
    arguments = parser.parse(sys.argv[1:])

    brainstem_file = arguments["-i"]
    segmented_file = arguments["-c"]
    output_file_name = arguments["-o"]


    label_depth_compared_to_zmax = arguments["-d"]
    label_value = arguments["-v"]
    #if "-d" in arguments:
        #label_depth_compared_to_zmax = arguments["-d"]
    #if "-v" in arguments:
        #label_value = arguments["-v"]



    add_label(brainstem_file, segmented_file, output_file_name, label_depth_compared_to_zmax, label_value)

    #Hardcode test:
#os.chdir('/home/tamag/data/template/errsm_35/t2')


#add_label("labels_brainstem.nii.gz", "data_RPI_seg.nii.gz", "labels_brainstem_completed", 10, 1)