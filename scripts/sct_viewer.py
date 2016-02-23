#!/usr/bin/env python
#########################################################################################
#
# Visualizer for MRI volumes
#
#
# ---------------------------------------------------------------------------------------
# Copyright (c) 2015 Polytechnique Montreal <www.neuro.polymtl.ca>
# Authors: Benjamin De Leener
# Created: 2015-01-30
# Modified: 2015-02-02
#
# About the license: see the file LICENSE.TXT
#########################################################################################
import sys
from msct_parser import Parser
from msct_image import Image


# from matplotlib.widgets import Slider, Button, RadioButtons


class TrioPlot:
    """
    This class manages mouse events on the three image subplots.
    """
    def __init__(self, fig_axial, fig_frontal, fig_sagittal, volume):
        self.fig_axial = fig_axial
        self.fig_frontal = fig_frontal
        self.fig_sagittal = fig_sagittal
        self.volume = volume

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.fig_axial.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig_axial.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig_axial.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

        self.cidpress = self.fig_frontal.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig_frontal.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig_frontal.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

        self.cidpress = self.fig_sagittal.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.fig_sagittal.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.fig_sagittal.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        self.press = event.xdata, event.ydata
        return

    def draw(self):
        self.fig_axial.figure.canvas.draw()
        self.fig_frontal.figure.canvas.draw()
        self.fig_sagittal.figure.canvas.draw()

    def move(self, event):
        if event.xdata and abs(event.xdata-self.press[0])<1 and abs(event.ydata-self.press[0])<1:
            self.press = event.xdata, event.ydata
            return

        if event.inaxes == self.fig_axial.axes:
            self.fig_frontal.set_data(self.volume.data[event.ydata,:,:])
            self.fig_sagittal.set_data(self.volume.data[:,:,event.xdata])

            self.fig_frontal.figure.canvas.draw()
            self.fig_sagittal.figure.canvas.draw()

            self.press = event.xdata, event.ydata
            return

        elif event.inaxes == self.fig_frontal.axes:
            self.fig_axial.set_data(self.volume.data[:,event.ydata,:])
            self.fig_sagittal.set_data(self.volume.data[:,:,event.xdata])

            self.fig_axial.figure.canvas.draw()
            self.fig_sagittal.figure.canvas.draw()

            self.press = event.xdata, event.ydata
            return

        elif event.inaxes == self.fig_sagittal.axes:
            self.fig_axial.set_data(self.volume.data[:,event.xdata,:])
            self.fig_frontal.set_data(self.volume.data[event.ydata,:,:])

            self.fig_axial.figure.canvas.draw()
            self.fig_frontal.figure.canvas.draw()

            self.press = event.xdata, event.ydata
            return

        else: return

    def on_motion(self, event):
        if event.button == 1:
            return self.move(event)
        else:
            return


    def on_release(self, event):
        if event.button == 1:
            return self.move(event)
        else:
            return

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.fig_axial.figure.canvas.mpl_disconnect(self.cidpress)
        self.fig_axial.figure.canvas.mpl_disconnect(self.cidrelease)
        self.fig_axial.figure.canvas.mpl_disconnect(self.cidmotion)

        self.fig_frontal.figure.canvas.mpl_disconnect(self.cidpress)
        self.fig_frontal.figure.canvas.mpl_disconnect(self.cidrelease)
        self.fig_frontal.figure.canvas.mpl_disconnect(self.cidmotion)

        self.fig_sagittal.figure.canvas.mpl_disconnect(self.cidpress)
        self.fig_sagittal.figure.canvas.mpl_disconnect(self.cidrelease)
        self.fig_sagittal.figure.canvas.mpl_disconnect(self.cidmotion)

class VolViewer(object):
    """
    This class is a visualizer for volumes (3D images).
    """
    def __init__(self,image):
        if isinstance(image,Image):
            self.image = image
        else:
            print "Error, the image is actually not an image"

    def updateAxial(self,val):
        self.im_plot_axial.set_data(self.image.data[:,val,:])
        self.fig.canvas.draw()

    def onclickAxial(self,event):
        if event.inaxes is not None:
            ax = event.inaxes
            print ax
            print event.x, event.y
            self.im_plot_frontal.set_data(self.image.data[event.y,:,:])
            self.im_plot_sagittal.set_data(self.image.data[:,:,event.x])
            self.fig.canvas.draw()

    def show(self):
        import matplotlib.pyplot as plt
        self.fig = plt.figure()
        self.fig.subplots_adjust(bottom=0.1, left=0.1)


        self.im_size = self.image.data.shape

        ax = self.fig.add_subplot(221)
        self.im_plot_axial = ax.imshow(self.image.data[:,int(self.im_size[1]/2),:])
        self.im_plot_axial.set_cmap('gray')
        self.im_plot_axial.set_interpolation('nearest')

        ax = self.fig.add_subplot(222)
        self.im_plot_frontal = ax.imshow(self.image.data[int(self.im_size[0]/2),:,:])
        self.im_plot_frontal.set_cmap('gray')
        self.im_plot_frontal.set_interpolation('nearest')

        ax = self.fig.add_subplot(223)
        self.im_plot_sagittal = ax.imshow(self.image.data[:,:,int(self.im_size[2]/2)])
        self.im_plot_sagittal.set_cmap('gray')
        self.im_plot_sagittal.set_interpolation('nearest')

        trio = TrioPlot(self.im_plot_axial, self.im_plot_frontal, self.im_plot_sagittal, self.image)
        trio.connect()

        #slider_ax = self.fig.add_axes([0.15, 0.05, 0.75, 0.03])
        #slider_axial = Slider(slider_ax, 'Axial slices', 0, self.im_size[1], valinit=int(self.im_size[1]/2))
        #slider_axial.on_changed(self.updateAxial)

        plt.show()



#=======================================================================================================================
# Start program
#=======================================================================================================================
if __name__ == "__main__":
    parser = Parser(__file__)
    parser.usage.set_description('Volume Viewer')
    parser.add_option("-i", "file", "file", True)
    arguments = parser.parse(sys.argv[1:])

    image = Image(arguments["-i"])
    viewer = VolViewer(image)
    viewer.show()
