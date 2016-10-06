#!/usr/bin/env python
#########################################################################################
#
# QcPatch class implementation
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Frederic Cloutier
# Modified: 2016-10-03
#
# About the license: see the file LICENSE.TXT
#########################################################################################
class QcPatch:

    def __init__(self,imageName, segImageName, contrast_type):
        self.imageName = imageName
        self.segImageName=segImageName
        self.contrast_type=contrast_type

    def savePatches(self):
        from msct_image import Image
        from scipy import ndimage
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        image = Image(self.imageName)
        nx, ny, nz, nt, px, py, pz, pt = image.dim
        seg = Image(self.segImageName)

        for i in range(nz):
            img = image.data[:, i, :]
            segImg = seg.data[:, i, :]
            size = 10;
            bc = ndimage.measurements.center_of_mass(segImg)
            x = int(round(bc[0]))
            y= int(round(bc[1]))
            mask = self.patchSlice(segImg, x, y,size)
            img = self.patchSlice(img, x, y, size)

            fig = plt.imshow(img, cmap='gray',interpolation='none')
            my_cmap = cm.hsv
            my_cmap.set_under('k', alpha=0) #how the color map deals with clipping values
            #  you can change your threshold in clim values
            plt.imshow(mask, cmap=my_cmap,interpolation='none',clim=[0.9, 1], alpha= 0.5)
            fig.axes.get_xaxis().set_visible(False)
            fig.axes.get_yaxis().set_visible(False)
            plt.savefig(self.contrast_type+'_sct_propseg_'+str(i)+'_withSeg.png', format='png',bbox_inches='tight', pad_inches = 0)
            #plt.imshow(mask, cmap=my_cmap, interpolation='none', clim=[0.9, 1], alpha=0.0)
            #fig.axes.get_xaxis().set_visible(False)
            #fig.axes.get_yaxis().set_visible(False)
            #plt.savefig(self.contrast_type + '_sct_propseg_' + str(i) + '_withoutSeg.png', format='png',
             #           bbox_inches='tight', pad_inches=0)

            plt.close()

    def patchSlice(self, matrix,centerX,centerY,ray):
        startRow = centerX -ray
        endRow = centerX+ray
        startCol =centerY-ray
        endCol = centerY + ray
        return [row[startCol:endCol] for row in matrix[startRow:endRow]]