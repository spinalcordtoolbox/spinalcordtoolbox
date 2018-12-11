# This is script will be moved elsewhere (not supposed to be present in sct)
# test symmetry detection method on sample images

import os
import numpy as np
from spinalcordtoolbox.image import Image
import matplotlib.pyplot as plt


from sct_axial_rotation import symmetry_angle, save_nifti_like, generate_2Dimage_line


triangle = np.array(Image("triangle.nii").data)
triangle2 = np.mean(np.array(Image("triangle2.nii").data), axis=2)
triangle3 = np.mean(np.array(Image("triangle3.nii").data), axis=2)

circle = np.mean(np.array(Image("circle.nii").data), axis=2)
circle2 = np.mean(np.array(Image("circle2.nii").data), axis=2)

square = np.array(Image("square.nii").data)

test = np.mean(np.array(Image("test.nii").data), axis=2)
test2 = np.mean(np.array(Image("test2.nii").data), axis=2)

nb_axes = 3

kmedian_size = 1

for image in [(triangle, "triangle.nii"), (triangle2, "triangle2.nii"), (triangle3, "triangle3.nii"),
              (circle, "circle.nii"), (circle2, "circle2.nii"), (square, "square.nii"), (test, 'test.nii'), (test2, 'test2.nii')]:

    angles = symmetry_angle(image[0], nb_axes=nb_axes, kmedian_size=kmedian_size)
    # centermass = image[0].mean(1).round().astype(int)  # will act weird if image is non binary
    centermass = [int(round(image[0].shape[0]/2)), int(round(image[0].shape[1]/2))]  # center of image

    image_wline = image[0]

    for i_angle in range(0, len(angles)):

        image_wline = generate_2Dimage_line(image_wline, centermass[0], centermass[1], angles[i_angle]-135)

    save_nifti_like(image_wline, "sym_" + image[1], "triangle.nii")




