#import numpy
import os
#from math import sqrt
#from scipy import ndimage

#import nibabel
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
from sct_nurbs_modif import *
import make_centerline as centerline
import sct_utils as sct
import sys
import linear_fitting
import getopt


class param:
    def __init__(self):
        self.remove = 0
        self.verbose = 1
        self.sigma = 3
        self.write = 0
        self.fitting_method = 'NURBS'
        self.nurbs_ctl_points = 0
        self.centerline = None
        self.warp = 1


def main():
    div = [3,5,7,9,11,13,15,19,23,25]
    nurbs_ctl_points = param.nurbs_ctl_points
    fitting_method = param.fitting_method
    sigma = param.sigma
    centerline = param.centerline
    s = 0
    d = 0
    exit = 0
    file_name = ''
    warp = param.warp

    try:
        opts, args = getopt.getopt(sys.argv[1:],'hi:M:n:d:s:c:v:w:r:h:')
    except getopt.GetoptError as err:
        print str(err)
        usage()
    for opt, arg in opts:
        if opt == '-h':
            usage()
        elif opt in ('-i'):
            file_name = arg
        elif opt in ('-M'):
            fitting_method = arg
        elif opt in ('-n'):
            nurbs_ctl_points = int(arg)
        elif opt in ('-d'):
            d = 1
            div = int(arg)
        elif opt in ('-s'):
            s = 1
            sigma = str(arg)
        elif opt in ('-c'):
            centerline = str(arg)
            fname_centerline = centerline
        elif opt in ('-v'):
            verbose = int(arg)
        elif opt in ('-w'):
            write = arg
        elif opt in ('-r'):
            remove = arg

    print 'file to be processed: ',file_name
    path, fname, ext_fname = sct.extract_fname(file_name)

    if not fitting_method == 'NURBS' and not fitting_method == 'polynomial' and not fitting_method == 'non_parametrique' and not fitting_method == 'smooth':
        usage()

    if centerline == None:
        # Generating centerline using propseg (Warning: be sure propseg work well with the input file)
        print 'Applying propseg to get the centerline as a binary image...'
        fname_centerline = fname+'_centerline'
        cmd = 'sct_propseg -i ' + file_name + ' -o . -t t2 -centerline-binary'
        sct.run(cmd)
    else:
        path, fname_centerline, ext_fname = sct.extract_fname(centerline)

    if s is not 0:
        print 'centerline smoothing...'
        fname_smooth = fname_centerline+'_smooth'
        print 'Gauss sigma: ', s
        cmd = 'fslmaths ' + fname_centerline + ' -s ' + str(s) + ' ' + fname_smooth + ext_fname
        sct.run(cmd)
        fname_centerline = fname_smooth+ext_fname
    else:
        fname_centerline = centerline

    if fitting_method == 'NURBS':
        if not d and not nurbs_ctl_points:
            for d in div:
                add = d - 1
                e = 1
                print 'generating the centerline...'

                # This loops stands for checking if nurbs will work with d
                while e == 1:
                    add += 1
                    e = centerline.check_nurbs(add, None, None, fname_smooth+ext_fname)
                    if add > 30:
                        exit = 1
                        break
                if exit == 1:
                    break
                d = add
                size = e

                nurbs_ctl_points = int(size)/d
        elif not nurbs_ctl_points:
            nurbs_ctl_points = int(size)/d

            print 'straightening...  d = ', str(d)

            # STRAIGHTEN USING NURBS
            cmd = 'sct_straighten_spinalcord -i ' + file_name + ' -c ' + fname_centerline + ' -n ' + str(nurbs_ctl_points)
            sct.run(cmd)

            print 'apply warp to segmentation'
            #final_file_name = fname+'_straightttt_seg'+ext_fname
            final_file_name = fname + '_straight_seg' + ext_fname
            #cmd = 'sct_WarpImageMultiTransform 3 '+fname_seg+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'
            cmd = 'sct_WarpImageMultiTransform 3 ' + fname_centerline + ' ' + final_file_name + ' warp_curve2straight.nii.gz'

            sct.run(cmd)

            print 'annalyzing the straightened file'
            linear_fitting.returnSquaredErrors(final_file_name, d, size)

    elif fitting_method == 'polynomial':
        # STRAIGHTEN USING POLYNOMIAL FITTING
        cmd = 'sct_straighten_spinalcord -i ' + file_name + ' -c ' + fname_smooth + ext_fname + ' -f polynomial -v 2'
        d = 'polynomial'
        size = 13
        # STRAIGHTEN USING 'non_parametric'
        sct.run(cmd)

        print 'apply warp to segmentation'
        #final_file_name = fname+'_straightttt_seg'+ext_fname
        final_file_name = fname+'_straight_seg'+ext_fname
        #cmd = 'sct_WarpImageMultiTransform 3 '+fname_seg+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'
        cmd = 'sct_WarpImageMultiTransform 3 '+fname_smooth+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'

        sct.run(cmd)

    elif fitting_method == 'smooth':
        # STRAIGHTEN USING POLYNOMIAL FITTING
        cmd = 'sct_straighten_spinalcord -i ' + file_name + ' -c ' + fname_centerline  + ' -f smooth -v 2'
        d = 'smooth'
        size = 13
        # STRAIGHTEN USING 'non_parametric'
        sct.run(cmd)

        if warp == 1:
            print 'apply warp to segmentation'
            #final_file_name = fname+'_straightttt_seg'+ext_fname
            final_file_name = fname+'_straight_seg'+ext_fname
            #cmd = 'sct_WarpImageMultiTransform 3 '+fname_seg+ext_fname+' '+final_file_name+' warp_curve2straight.nii.gz'
            cmd = 'sct_WarpImageMultiTransform 3 '+fname_centerline+' '+final_file_name+' warp_curve2straight.nii.gz'
        else:
            print 'Applying propseg to the straightened volume...'
            fname_straightened = fname+'_straight'
            cmd = 'sct_propseg -i ' + fname_straightened + ext_fname +' -o . -t t2 -centerline-binary'
            final_file_name = fname_straightened + '_centerline' + ext_fname

        sct.run(cmd)

    elif fitting_method == 'non-parametric':
        cmd = 'sct_straighten_spinalcord -i '+file_name+' -c '+fname_smooth+ext_fname+' -f non_parametrique -v 2'
        d = 'polynomial'
        size = 13
        # STRAIGHTEN USING 'non_parametric'
        sct.run(cmd)

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


def usage():
    print '\n' \
        ''+os.path.basename(__file__)+'\n' \
        '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n' \
        'Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>\n' \
        '\n'\
        'DESCRIPTION\n' \
        ' Script for estimate efficiency of a fitting method in sct_sctraighten_spinnalcord for a particular file.' \
        ' Currently returns MSE between final straigntened centerline and the expected one.\n' \
        '\n' \
        'Fitting methods:' \
        '   NURBS' \
        '   non-parametric' \
        '   polynomial'\
        'USAGE\n' \
        '  '+os.path.basename(__file__)+' -i <file> -M <method>\n' \
        '\n'\
        'MANDATORY ARGUMENTS\n' \
        '  -i                input volume.\n' \
        '  -M                method (NURBS, non-parametric, polynomial)' \
        '\n'\
        'OPTIONAL ARGUMENTS\n' \
        '  -n <nbctlpoints>      only when choosing NURBS method, set the nb of control points. Default is a fraction of the spinnal cord length of the volume you set as input.\n' \
        '  -d <div>              only when choosing NURBS method, set the nb of control points as spinalcord length / div.\n' \
        '  -s <sigma>            set the sigma used for smoothing the centerline. Set <sigma> = 0 if you do not want your centerline to be smoothed. Default is ' + str(param.sigma) + '\n' \
        '  -c <centerline>       Skip the centerline automatic generation and smoothing and use <centerline>. Default is generated by sct_propseg and smoothed with fslmaths.\n' \
        '  -v {0,1,2}            verbose. 0: nothing, 1: txt, 2: txt+fig. Default=' + str(param.verbose) + '\n' \
        '  -w                    write results when list of test is scheduled (ie. parameters for the different methods)' \
        '  -r {0,1}              remove temporary file, default is ' + str(param.remove) + '\n' \
        '  -h                    help. Show this message.\n' \
        '\n'
    sys.exit(2)


if __name__ == "__main__":
    #print 'Number of arguments:', len(sys.argv), 'arguments.'
    param = param()
    main()


'''
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
    '''