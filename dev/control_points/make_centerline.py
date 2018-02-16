# Main fonction return the centerline of the mifti image fname as a nifti binary file
# Centerline is generated using sct_nurbs with nbControl = size/div


from sct_nurbs_v2 import *
import nibabel
import splines_approximation_v2 as spline_app
from scipy import ndimage
import numpy

import linear_fitting as lf
import sct_utils


def returnCenterline(fname = None, nurbs = 0, div = 0):


    if fname == None:
        fname = 't250_half_sup_straight_seg.nii.gz'


    file = nibabel.load(fname)
    data = file.get_data()
    hdr_seg = file.get_header()


    nx, ny, nz = spline_app.getDim(fname)
    x = [0 for iz in range(0, nz, 1)]
    y = [0 for iz in range(0, nz, 1)]
    z = [iz for iz in range(0, nz, 1)]
    for iz in range(0, nz, 1):
            x[iz], y[iz] = ndimage.measurements.center_of_mass(numpy.array(data[:,:,iz]))
    points = [[x[n],y[n],z[n]] for n in range(len(x))]

    p1, p2, p3 = spline_app.getPxDimensions(fname)
    size = spline_app.getSize(x, y, z, p1, p2, p3)

    data = data*0

    if nurbs:
        if check_nurbs(div, size, points) != 0:
            x_centerline_fit=P[0]
            y_centerline_fit=P[1]
            z_centerline_fit=P[2]
            for i in range(len(z_centerline_fit)) :
                data[int(round(x_centerline_fit[i])),int(round(y_centerline_fit[i])),int(z_centerline_fit[i])] = 1
        else: return 1
    else:
        for i in range(len(z)) :
            data[int(round(x[i])),int(round(y[i])),int(z[i])] = 1


    path, file_name, ext_fname = sct_utils.extract_fname(fname)
    img = nibabel.Nifti1Image(data, None, hdr_seg)

    #return img

    saveFile(file_name, img, div)
    return size


def check_nurbs(div, size = 0, points = 0, centerline = ''):
    if centerline == '':
        print 'div = ',div,' size = ', round(size)
        nurbs = NURBS(int(round(size)), int(div), 3, 3000, points)
        P = nurbs.getCourbe3D()
        if P==1:
            print "ERROR: instability in NURBS computation, div will be incremented. "
            return 1

    else:
        file = nibabel.load(centerline)
        data = file.get_data()
        hdr_seg = file.get_header()
        nx, ny, nz = spline_app.getDim(centerline)

        x = [0 for iz in range(0, nz, 1)]
        y = [0 for iz in range(0, nz, 1)]
        z = [iz for iz in range(0, nz, 1)]
        for iz in range(0, nz, 1):
                x[iz], y[iz] = ndimage.measurements.center_of_mass(numpy.array(data[:,:,iz]))
        points = [[x[n],y[n],z[n]] for n in range(len(x))]

        p1, p2, p3 = spline_app.getPxDimensions(centerline)
        size = spline_app.getSize(x, y, z, p1, p2, p3)

        print 'div = ',div,' size = ', round(size)

        #nurbs = NURBS(int(round(size)), int(div), 3, 3000, points)      --> this work with sct_nurbs_v1
        try:
            nurbs = NURBS(3, 3000, points, False, None, int(round(size)), int(div))
            P = nurbs.getCourbe3D()
        except UnboundLocalError:
            print "ERROR: instability in NURBS computation, UnboundLocalError caught, div will be incremented. "
            return 1
        except ZeroDivisionError:
            print "ERROR: instability in NURBS computation, ZeroDivisionError caught, div will be incremented. "
            return 1

        if P==1:
            print "ERROR: instability in NURBS computation, div will be incremented. "
            return 1
        else: return round(size)


def saveFile(file_name, img, div):
	path_centerline = './centerlines/'+file_name+'_'+str(div)+'_centerline.nii.gz'
	nibabel.save(img,path_centerline)

	#cmd = 'sct_straighten_spinalcord -i '+path_centerline+' -c '+fname
	#print cmd
	#sct.run(cmd)
	#cmd = 'sct_propseg'


if __name__ == "__main__":
    returnCenterline()