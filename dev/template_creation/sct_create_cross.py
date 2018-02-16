#!/usr/bin/env python
#########################################################################################
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2014 Polytechnique Montreal <www.neuro.polymtl.ca>
# Author: Julien Touati
# Created: 2014-08-11
#
# About the license: see the file LICENSE.TXT
#########################################################################################

#DEFAULT PARAMETERS
class param:
    ## The constructor
    def __init__(self):
        self.verbose = 1
        self.gapxy = 15.0  # millimeters or voxel
        self.cross = 'mm'
# check if needed Python libraries are already installed or not
import sys
import getopt

# Get path of the toolbox
path_sct = os.environ.get("SCT_DIR", os.path.dirname(os.path.dirname(__file__)))
# Append path that contains scripts, to be able to load modules
sys.path.append(os.path.join(path_sct, "scripts"))


import sct_utils as sct
import nibabel
import os

def main():
    
    #Initialization
    fname = ''
    fname_template = ''
    verbose = param.verbose
    gapxy = param.gapxy
    cross = param.cross
    x = ''
    y = ''
    zmin = ''
    zmax = ''

    try:
         opts, args = getopt.getopt(sys.argv[1:],'hi:x:y:s:e:c:t:v:')
    except getopt.GetoptError:
        usage()
    for opt, arg in opts :
        if opt == '-h':
            usage()
        elif opt in ("-i"):
            fname = arg
        elif opt in ('-x'):
            x = int(arg)
        elif opt in ('-y'):
            y = int(arg)
        elif opt in ('-s'):
            zmin = int(arg)
        elif opt in ('-e'):
            zmax = int(arg)
        elif opt in ('-c'):
            cross = arg    
        elif opt in ("-t"):
            fname_template = arg
        elif opt in ('-v'):
            verbose = int(arg)
    
    # display usage if a mandatory argument is not provided
    if fname == '' and fname_template == '':
        usage()
    
    # check existence of input files
    print'\nCheck if file exists ...'
    
    if fname != '':
        sct.check_file_exist(fname)
    
    if fname_template != '':
        sct.check_file_exist(fname_template)
        
    if cross not in ['mm','voxel'] :
        usage()    
    
    # Display arguments
    print'\nCheck input arguments...'
    print'  Input volume ...................... '+fname
    print'  Template ...................... '+fname_template
    print'  Verbose ........................... '+str(verbose)
    
    if fname != '':
        print '\nGet dimensions of input...'
        from msct_image import Image
        nx, ny, nz, nt, px, py, pz, pt = Image(fname).dim
        print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
        print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
    
        file = nibabel.load(fname)
        data = file.get_data()
        hdr = file.get_header()
    
        data *= 0

        list_opts = []
        for i in range(len(opts)):
            list_opts.append(opts[i][0])

        if cross == 'mm' :
            gapx = int(round(gapxy/px))
            gapy = int(round(gapxy/py))

            if ("-s") in list_opts and ("-e") in list_opts and zmax < nz:
                data[x,y,zmin] = 1
                data[x,y,zmax] = 2
                data[x+gapx,y,zmax] = 3
                data[x-gapx,y,zmax] = 4
                data[x,y+gapy,zmax] = 5
                data[x,y-gapy,zmax] = 6
            else:
                data[x,y,0] = 1
                data[x,y,nz-1] = 2
                data[x+gapx,y,nz-1] = 3
                data[x-gapx,y,nz-1] = 4
                data[x,y+gapy,nz-1] = 5
                data[x,y-gapy,nz-1] = 6
    
        if cross == 'voxel' :
            gapxy = int(gapxy)

            if ("-s") in list_opts and ("-e") in list_opts and zmax < nz:
                data[x,y,zmin] = 1
                data[x,y,zmax] = 2
                data[x+gapx,y,zmax] = 3
                data[x-gapx,y,zmax] = 4
                data[x,y+gapy,zmax] = 5
                data[x,y-gapy,zmax] = 6
            else:
                data[x,y,0] = 1
                data[x,y,nz-1] = 2
                data[x+gapx,y,nz-1] = 3
                data[x-gapx,y,nz-1] = 4
                data[x,y+gapy,nz-1] = 5
                data[x,y-gapy,nz-1] = 6
    
        print '\nSave volume ...'
        hdr.set_data_dtype('float32') # set imagetype to uint8
        # save volume
        #data = data.astype(float32, copy =False)
        img = nibabel.Nifti1Image(data, None, hdr)
        file_name = 'landmark_native.nii.gz'
        nibabel.save(img,file_name)
        print '\nFile created : ' + file_name
    
    if fname_template != '':
    
        print '\nGet dimensions of template...'
        nx, ny, nz, nt, px, py, pz, pt = Image(fname_template).dim
        print '.. matrix size: '+str(nx)+' x '+str(ny)+' x '+str(nz)
        print '.. voxel size:  '+str(px)+'mm x '+str(py)+'mm x '+str(pz)+'mm'
        
        file_t = nibabel.load(fname_template)
        data_t = file_t.get_data()
        hdr_t = file_t.get_header()

        data_t *= 0
        
        if cross == 'mm':
            
            gapx = int(round(gapxy/px))
            gapy = int(round(gapxy/py))

            if ("-s") in list_opts and ("-e") in list_opts and zmax < nz:
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),zmin] = 1
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),zmax] = 2
                data_t[int(round(nx/2.0)) + gapx,int(round(ny/2.0)),zmax] = 3
                data_t[int(round(nx/2.0)) - gapx,int(round(ny/2.0)),zmax] = 4
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) + gapy,zmax] = 5
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) - gapy,zmax] = 6

            else:
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),0] = 1
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),nz-1] = 2
                data_t[int(round(nx/2.0)) + gapx,int(round(ny/2.0)),nz-1] = 3
                data_t[int(round(nx/2.0)) - gapx,int(round(ny/2.0)),nz-1] = 4
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) + gapy,nz-1] = 5
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) - gapy,nz-1] = 6
 
        if cross == 'voxel':
            
            gapxy = int(gapxy)

            # data_t[int(round(nx/2.0)),int(round(ny/2.0)),0] = 1
            # data_t[int(round(nx/2.0)),int(round(ny/2.0)),nz-1] = 2
            # data_t[int(round(nx/2.0)) + gapxy,int(round(ny/2.0)),nz-1] = 3
            # data_t[int(round(nx/2.0)) - gapxy,int(round(ny/2.0)),nz-1] = 4
            # data_t[int(round(nx/2.0)),int(round(ny/2.0)) + gapxy,nz-1] = 5
            # data_t[int(round(nx/2.0)),int(round(ny/2.0)) - gapxy,nz-1] = 6

            if ("-s") in list_opts and ("-e") in list_opts and zmax < nz:
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),zmin] = 1
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),zmax] = 2
                data_t[int(round(nx/2.0)) + gapxy,int(round(ny/2.0)),zmax] = 3
                data_t[int(round(nx/2.0)) - gapxy,int(round(ny/2.0)),zmax] = 4
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) + gapxy,zmax] = 5
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) - gapxy,zmax] = 6

            else:
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),0] = 1
                data_t[int(round(nx/2.0)),int(round(ny/2.0)),nz-1] = 2
                data_t[int(round(nx/2.0)) + gapxy,int(round(ny/2.0)),nz-1] = 3
                data_t[int(round(nx/2.0)) - gapxy,int(round(ny/2.0)),nz-1] = 4
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) + gapxy,nz-1] = 5
                data_t[int(round(nx/2.0)),int(round(ny/2.0)) - gapxy,nz-1] = 6
            
        print '\nSave volume ...'
        hdr_t.set_data_dtype('float32') # set imagetype to uint8
        # save volume
        #data = data.astype(float32, copy =False)
        img_t = nibabel.Nifti1Image(data_t, None, hdr_t)
        file_name_t = 'template_landmarks.nii.gz'
        nibabel.save(img_t,file_name_t)
        print '\nFile created : ' + file_name_t

    
def usage():
    print """
"""+os.path.basename(__file__)+"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Part of the Spinal Cord Toolbox <https://sourceforge.net/projects/spinalcordtoolbox>

DESCRIPTION

  Create a cross at the top and bottom of the volume (i.e., Zmin and Zmax). This script assumes that
  the input image is a cropped straighten spinalcord volume. Does the same for the template if inputed.
  Zmin and zmax can also be specified if needed.

USAGE
  """+os.path.basename(__file__)+"""  -i <input_volume> -x <x_centerline> -y <y_centerline>

MANDATORY ARGUMENTS
  -i <input_volume>         input straight cropped volume. No Default value
  -x                        x coordinate of the centerline. See sct_detect_extrema.py
  -y                        y coordinate of the centerline. See sct_detect_extrema.py
 
OPTIONAL ARGUMENTS
  -s                         z bottom coordinate
  -e                         z top coordinate
  -v {0,1}                   verbose. Default="""+str(param.verbose)+"""
  -t <template>              template if you want to create landmarks in 
                             the template space
  -c {voxel,mm}              'voxel' creates cross of same size in terms of voxels in
                             both the anatomical and template image
                             'mm' creates cross of same size in terms of mm in both 
                             the anatomical and template image
                             Default = """+param.cross+"""
  -h                         help. Show this message

EXAMPLE
  """+os.path.basename(__file__)+""" -i straight_centerline.nii.gz -x 55 -y 222\n"""

    # exit program
    sys.exit(2)


#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    # initialize parameters
    param = param()
    # call main function
    main()



