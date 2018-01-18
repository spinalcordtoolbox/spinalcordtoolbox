import numpy
import os
from math import sqrt
from scipy import ndimage

import nibabel
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sct_nurbs_modif import *
import make_centerline as centerline
import sct_utils as sct
import sys
import linear_fitting


def main(file_name, smooth = 3):
    div = [25]
    #div = [13,15,19,23,25]
    #div = [5,7,9,11,13,15,19,23,25]
    #div = int(div)
    path, fname, ext_fname = sct.extract_fname(file_name)
    exit = 0

    print 'file to be processed: ',file_name

    print 'Applying propseg to get the centerline as a binary image...'
    #fname_seg = fname+'_seg'
    fname_centerline = fname+'_centerline'
    cmd = 'sct_propseg -i '+file_name+' -o . -t t2 -centerline-binary'
    sct.run(cmd)


    print 'centerline smoothing...'
    fname_smooth = fname_centerline+'_smooth'
    print 'Gauss sigma: ', smooth
    cmd = 'fslmaths '+fname_centerline+' -s '+str(smooth)+' '+fname_smooth+ext_fname
    sct.run(cmd)


    for d in div:
        add = d - 1
        e = 1
        print 'generating the centerline...'
        while e == 1:
            add += 1
            #e = centerline.returnCenterline(fname_smooth+ext_fname, 1, add)
            e = centerline.check_nurbs(add, None, None, fname_smooth+ext_fname)
            if add > 30:
                exit = 1
                break

        if exit == 1:
            break
        d = add
        size = e
        nurbs_ctl_points = int(size)/d

        #d = returnCenterline(fname_smooth+ext_fname, d)



        print 'straightening...  d = ', str(d)
        #fcenterline = './centerlines/'+fname_smooth+'_'+str(d)+'_centerline.nii.gz'
        cmd = 'sct_straighten_spinalcord -i '+file_name+' -c '+fname_smooth+ext_fname+' -n '+str(nurbs_ctl_points)
        sct.run(cmd)

        '''
        print 'applying propseg'
        fname_straight = fname+'_straight'+ext_fname
        final_file_name = fname+'_straight_seg'+ext_fname
        cmd = 'sct_propseg -i '+fname_straight+' -t t2'

        '''

        print 'apply warp to segmentation'
        #final_file_name = fname+'_straightttt_seg'+ext_fname
        final_file_name = fname+'_straight_seg'+ext_fname
        #cmd = 'sct_WarpImageMultiTransform 3 '+fname_seg+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'
        cmd = 'sct_WarpImageMultiTransform 3 '+fname_smooth+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'

        sct.run(cmd)
        

        print 'annalyzing the straightened file'
        linear_fitting.returnSquaredErrors(final_file_name, d, size)

    os.remove(fname_centerline+ext_fname)
    os.remove(fname_smooth+ext_fname)
    os.remove('warp_curve2straight.nii.gz')
    os.remove('warp_straight2curve.nii.gz')

def returnCenterline(fname, div):
    if centerline.returnCenterline(fname, 1, div) == 1:#this save the centerline: centerline/file_name_div_centerline.nii.gz
        div += 1
        returnCenterline(fname, div)
    else:
        #print centerline.returnCenterline(fname, 1, div)
        #print div
        return div


    
if __name__ == "__main__":
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    #print 'Argument List:', str(sys.argv)
    file_name = sys.argv[1]
    #div = sys.argv[2]
    print file_name
    main(file_name)