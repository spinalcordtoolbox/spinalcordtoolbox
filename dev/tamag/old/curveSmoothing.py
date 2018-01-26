#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
from math import pi, cos
import matplotlib.pyplot as plt

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))

from msct_image import Image
from msct_types import Coordinate
from msct_parser import Parser
from sct_label_utils import ProcessLabels
import os
import sct_utils as sct


def curve_smoothing(file_seg_or_centerline, output_file_name, number_mm):   #voir a ajouter un parametre sur le nbr de mm

    image_to_modify=Image(file_seg_or_centerline).copy()

    im_output=Image(file_seg_or_centerline).copy()
    im_output.data *=0

    nx, ny, nz, nt, px, py, pz, pt = sct.get_dimension(file_seg_or_centerline)
    #nx: nbr de pixels selon x
    #px: taille d'un pixel en mm (mm)

    #Def of the window (hamming):  same size as the height of the image
    window_hamming=np.zeros(nz)
    for i in range(0,nz):
        window_hamming[i]=0.54-0.46*cos((2*pi*i)/nz)

    window_rectangular=np.zeros(nz)
    for i in range(0, nz):
        if i in range(nz-8, nz+8):
            window_rectangular[i]=1

    #Def of the window (hamming):  5mm  avec 50 points
    window_hamming_local=np.zeros(10*int(number_mm/pz))    #5mm  5mm=px*(nb_pixels_for_5mm)
    for i in range(0, int(number_mm/pz)):
        for j in range(0,10):
            point=i+j/10.0
            print(point)
            window_hamming_local[point*10]=0.54-0.46*cos((2*pi*point)/(int(number_mm/pz)))

    plt.figure(2)
    plt.plot(window_hamming_local[:])
    plt.show()

    #Definition d'un vecteur "coordi" qui va regrouper le resultat du smoothing selon z (fait reference aux coordonnees selon x ou y)
    coordi_x=np.zeros(nz)     #y "fixe"
    coordi_y=np.zeros(nz)     #x "fixe"

    print("nz=", nz)
    print("pz=", pz, "int(",number_mm,"/pz)=", int(number_mm/pz))
    print(5*3.5, float(5*3.5))

#Pour un recalcul des coordonnees en appliquant la fenêtre au voisinage
    X, Y, Z = np.nonzero((image_to_modify.data[:,:,:] > 0))   #

    print("Z[:]=",Z[:], "max(Z[:])=",max(Z[:]), "min(Z[:])=",min(Z[:]))


    plt.figure(1)
    plt.subplot(211)
    plt.plot(Z[:], X[:], 'ro')


    print("X.shape[0]=", X.shape[0], "Z.shape[0]=", Z.shape[0])
    if not X.any:
            print("The binary file is empty; no labels can be found.")
    for z in range(0, nz):      #z position du point considere (celui qui va recevoir la somme ponderee)
        #Calcul de la position des points non nuls de l'image
        #for i in range(0, int(number_mm/pz)):   #i ecart du point pondere au point qui recoit la somme
        for i in range(-5, 5):
            coordi_x[z] += window_hamming_local[int((int(number_mm/pz)/2.0)+i*10)%(int(number_mm/pz)*10)] * X[(z+i)%X.shape[0]]    #ajuster 10 en fonction du nbr de mm
            #coordi_x[z] += X[(z+i)%X.shape[0]]/5.0

            coordi_y[z] += window_hamming_local[int((int(number_mm/pz)/2.0)+i*10)%(int(number_mm/pz)*10)] * Y[(z+i)%Y.shape[0]]
        #print("coordi_x[z]=", coordi_x[z])

    plt.subplot(212)
    plt.plot(range(0, nz), coordi_x[:], 'ro')
    plt.show()

#        for x in range(0, image_to_modify.data.shape[0]):
#            for y in range(0, image_to_modify.data.shape[1]):
#                im_output[coordi,y,z]=
#
#    im_output=Image(file_seg_or_centerline).copy()
#    im_output.dat*=0

#Save output file
#    im_output.setFileName(output_file_name)   #is it finished? does the function need to return something?
#    im_output.save('minimize')




os.chdir("/home/tamag/data/template/errsm_35/t2")
curve_smoothing("data_RPI_centerline.nii.gz", "data_RPI_centerline_smoothed.nii.gz", 10)
