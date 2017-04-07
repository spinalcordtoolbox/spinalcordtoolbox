import sys
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore



class HeaderCore(object):

    def __init__(self):
        self.define_lb_status()
        self.define_lb_warning()
        self.define_layout_header()

    def define_lb_status(self):
        self.lb_status = QtGui.QLabel('Label Alerte')
        self.lb_status.setContentsMargins(10, 10, 10, 0)
        self.lb_status.setAlignment(QtCore.Qt.AlignCenter)

    def define_lb_warning(self):
        self.lb_warning = QtGui.QLabel('Label Warning')
        self.lb_warning.setContentsMargins(10, 10, 10, 10)
        self.lb_warning.setAlignment(QtCore.Qt.AlignCenter)

    def define_layout_header(self):
        self.layout_header = QtGui.QVBoxLayout()
        self.layout_header.setAlignment(QtCore.Qt.AlignTop)
        self.layout_header.addWidget(self.lb_status)
        self.layout_header.addWidget(self.lb_warning)

class Header(HeaderCore):

    def update_lb(self,key):
        if(key=='start'):
            self.lb_status.setText('header.lb_status')
            self.lb_warning.setText('header.lb_warning')
            self.lb_warning.setStyleSheet("color:red")



def launch_main_window():
    system = QtGui.QApplication(sys.argv)
    w = QtGui.QWidget()
    w.resize(740, 850)
    w.setWindowTitle('Hello world')
    w.show()
    return (w,system)

def add_layout_main(w):
    layout_main=QtGui.QVBoxLayout()
    layout_main.setAlignment(QtCore.Qt.AlignTop)
    w.setLayout(layout_main)
    return layout_main


def add_header(w):
    header=Header()
    w.addLayout(header.layout_header)
    header.update_lb('start')
    return(header)

def add_main_anat_view(layout_central):
    layout_anat_view = QtGui.QVBoxLayout()
    layout_anat_view.setAlignment(QtCore.Qt.AlignTop)
    layout_anat_view.setAlignment(QtCore.Qt.AlignRight)

    layout_anat_view.addWidget(create_image())
    layout_central.addLayout(layout_anat_view)

def add_secondary_anat_view(layout_central):
    layout_anat_view = QtGui.QVBoxLayout()
    layout_anat_view.setAlignment(QtCore.Qt.AlignTop)
    layout_anat_view.setAlignment(QtCore.Qt.AlignRight)

    layout_anat_view.addWidget(create_image())
    layout_central.addLayout(layout_anat_view)

def create_image():
    image_label=QtGui.QLabel('')
    image_test=QtGui.QPixmap('/home/apopov/Documents/dev/sct/image_test.jpg')
    image_label.setPixmap(image_test)
    return image_label

def add_central_layout(layout_main):
    layout_central=QtGui.QHBoxLayout()
    add_main_anat_view(layout_central)
    add_secondary_anat_view(layout_central)

    layout_main.addLayout(layout_central)


(window,system) = launch_main_window()

layout_main = add_layout_main(window)
header = add_header(layout_main)
layout_central = add_central_layout(layout_main)


window.setLayout(layout_main)


sys.exit(system.exec_())


