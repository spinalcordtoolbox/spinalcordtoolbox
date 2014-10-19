
Viewer for 4D data
==================

to try:
http://stackoverflow.com/questions/6697259/interactive-matplotlib-plot-with-two-sliders

~~~
import numpy as np
import pylab

class plotter:
    def __init__(self, initial_values):
        self.values
        self.fig = pylab.figure()
        pylab.gray()
        self.ax = self.fig.add_subplot(111)
        self.draw()
        self.fig.canvas.mpl_connect('key_press_event',self.key)

    def draw(self):
        im = your_function(self.values)
        pylab.show()
        self.ax.imshow(im)

    def key(self, event):
        if event.key=='right':
            self.values = modify()
        elif event.key == 'left':
            self.values = modify()

        self.draw()
        self.fig.canvas.draw()
~~~
