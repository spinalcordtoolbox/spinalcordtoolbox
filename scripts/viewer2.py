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

class MainPannelCore(object):

    def __init__(self):
        self.layout_global=QtGui.QVBoxLayout()
        self.layout_option_settings = QtGui.QHBoxLayout()
        self.layout_central = QtGui.QHBoxLayout()

    def add_main_anat_view(self):
        layout_anat_view = QtGui.QVBoxLayout()
        layout_anat_view.setAlignment(QtCore.Qt.AlignTop)
        layout_anat_view.setAlignment(QtCore.Qt.AlignRight)

        layout_anat_view.addWidget(self.create_image())
        self.layout_central.addLayout(layout_anat_view)

    def add_secondary_anat_view(self):
        layout_anat_view = QtGui.QVBoxLayout()
        layout_anat_view.setAlignment(QtCore.Qt.AlignTop)
        layout_anat_view.setAlignment(QtCore.Qt.AlignRight)

        layout_anat_view.addWidget(self.create_image())
        self.layout_central.addLayout(layout_anat_view)

    def create_image(self):
        image_label = QtGui.QLabel('')
        image_test = QtGui.QPixmap('/home/apopov/Documents/dev/sct/image_test.jpg')
        image_label.setPixmap(image_test)
        return image_label

    def merge_layouts(self):
        self.layout_global.addLayout(self.layout_option_settings)
        self.layout_global.addLayout(self.layout_central)

class MainPannel(MainPannelCore):

    def __init__(self):
        super(MainPannel, self).__init__()
        self.add_main_anat_view()
        self.add_secondary_anat_view()

        self.merge_layouts()

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

def add_main_pannel(layout_main):
    mainPannel=MainPannel()
    layout_main.addLayout(mainPannel.layout_global)


(window,system) = launch_main_window()

layout_main = add_layout_main(window)
header = add_header(layout_main)
main_pannel = add_main_pannel(layout_main)


window.setLayout(layout_main)


sys.exit(system.exec_())


