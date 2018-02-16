#!/usr/bin/env python

import sys, io, os, random, math
from time import time

import numpy as np
import nibabel as nib
import matplotlib as plt

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, 'scripts'))

import sct_utils as sct

def aleaGauss(sigma):
    U1 = random.random()
    U2 = random.random()
    return sigma*math.sqrt(-2*math.log(U1))*math.cos(2*math.pi*U2)

def add_noise_gaussian(file_to_noise):

    img = nib.load(file_to_noise)
    hdr_0 = img.get_header()
    data = img.get_data()

    path, file, ext = sct.extract_fname(file_to_noise)

    s=data.shape
    sigma=10

    t = time()

    for j in range(s[0]):
        for i in range(s[1]):
            for k in range(s[2]):
                v = int(math.floor(data[j][i][k]+random.gauss(0,sigma)))
                if v > 2**16:
                    v = 2**16
                if v<0:
                    v = 0
                data[j][i][k] = v

    print("total time", time() - t)
    print("vol size", data.shape)

    img_noise = nib.Nifti1Image(data, None, hdr_0)
    nib.save(img_noise, file + '_noise' +ext)



os.chdir("/Users/tamag/data/original_data/full/errsm_03/t2")
add_noise_gaussian("data_RPI.nii.gz")
