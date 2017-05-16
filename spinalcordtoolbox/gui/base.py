import sys
import webbrowser
from copy import copy
from time import time
import os

import PyQt4.QtCore as QtCore
import PyQt4.QtGui as QtGui
import matplotlib as mpl
import matplotlib.pyplot as plt
import sct_utils as sct
from matplotlib import cm
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.lines import Line2D
from msct_image import Image
from msct_types import Coordinate
from numpy import pad, percentile
