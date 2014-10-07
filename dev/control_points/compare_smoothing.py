import nibabel as nib
import splines_approximation_v2 as splinapp
from scipy import ndimage
import sct_utils as sct
import matplotlib.pyplot as plt
import sys
import numpy as np

#AJPUTER LE PLOT DES TOUTES LES COURBES


def main(fname):

    #list_sigma = [0.5, 5, 10, 15, 20]
    list_sigma = [0.5, 5, 10, 15, 20]

    for sigma in list_sigma:
        fname_smooth = smooth(fname, sigma)

        file = nib.load(fname_smooth)
        data = file.get_data()

        nx, ny, nz = splinapp.getDim(fname_smooth)

        x = [0 for iz in range(0, nz, 1)]
        y = [0 for iz in range(0, nz, 1)]
        z = [iz for iz in range(0, nz, 1)]

        for iz in range(0, nz, 1):
            x[iz], y[iz] = ndimage.measurements.center_of_mass(np.array(data[:, :, iz]))



        ax = plt.subplot(1,2,1)
        if sigma == 0.5 or sigma == 20:
            plt.plot(x, z, 'r-', label='sigma = ' + str(sigma))
            ax = plt.subplot(1,2,2)
            plt.plot(y, z, 'r-', label='sigma = ' + str(sigma))
            plt.xlabel('z')
            plt.ylabel('x')
        else:
            plt.plot(x, z, 'b-', label='sigma = ' + str(sigma))
            ax = plt.subplot(1,2,2)
            plt.plot(y, z, 'b-', label='sigma = ' + str(sigma))
            plt.xlabel('z')
            plt.ylabel('y')

    plt.show()


def smooth(fname, sigma):

    path, fname, ext_fname = sct.extract_fname(fname)

    print 'centerline smoothing...'
    fname_smooth = fname +'_smooth'
    print 'Gauss sigma: ', smooth
    cmd = 'fslmaths ' + fname + ' -s ' + str(sigma) + ' ' + fname_smooth + ext_fname
    sct.run(cmd)
    return fname_smooth + ext_fname


if __name__ == "__main__":
    file_name = sys.argv[1]
    print file_name
    main(file_name)