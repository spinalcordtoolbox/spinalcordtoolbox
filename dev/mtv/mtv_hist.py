__author__ = 'slevy'

import nibabel as nib
import numpy as np
import pylab as plt

fa_list = [5, 10, 20, 30]
for i in fa_list:
    mtv = nib.load('/home/django/slevy/data/handedness_asymmetries/errsm_31/mtv_investig/mtv_'+str(i)+'_crop.nii.gz').get_data()
    mask_csf = nib.load('/home/django/slevy/data/handedness_asymmetries/errsm_31/mtv_investig/mtv_'+str(i)+'_labels/mtv_'+str(i)+'_crop_csf.nii.gz').get_data()

    # histogram total
    mtv_csf_all = mtv[mask_csf == 1]

    fig = plt.figure(int(str(i)+'0'))
    fig.suptitle('Histogram of the signal of SPGR image with flip angle = '+str(i)+' degree in CSF', fontsize=16)
    ax = fig.add_subplot(111)

    ax.hist(mtv_csf_all, bins=50)
    plt.annotate('Mean +/- std = ' + str(np.mean(mtv_csf_all)) + ' +/- ' + str(np.std(mtv_csf_all)), xy=(0.3, 45), fontsize=16)
    ax.set_xlabel('Value of the signal', fontsize=16)
    ax.axis([0, 300, 0, 50])

    plt.show(block=False)

    mtv_per_slice = np.empty([mtv.shape[2]], dtype=object)
    for z in range(0, mtv.shape[2]):

        ind_csf_slice = mask_csf[:,:,z]==1

        mtv_per_slice[z] = mtv[ind_csf_slice, z]

        fig = plt.figure(int(str(i) + '0'+str(z)))
        fig.suptitle('Histogram of the signal of SPGR image with flip angle = ' + str(i) + ' degree in CSF (slice #'+str(z)+')', fontsize=16)
        ax = fig.add_subplot(111)

        ax.hist(mtv_per_slice[z], bins=20)
        plt.annotate('Mean +/- std = ' + str(np.mean(mtv_per_slice[z])) + ' +/- ' + str(np.std(mtv_per_slice[z])), xy=(0.3, 14),
                     fontsize=16)
        ax.set_xlabel('Value of the signal', fontsize=16)
        ax.axis([0, 300, 0, 15])

        plt.show(block=False)




plt.show()

