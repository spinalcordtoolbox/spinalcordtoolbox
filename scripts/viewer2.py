import sys
import PyQt4.QtGui as QtGui

def launch_main_window():
    system = QtGui.QApplication(sys.argv)
    w = QtGui.QWidget()
    w.resize(740, 850)
    w.setWindowTitle('Hello World')
    w.show()
    return (w,system)

def create_header(w):
    layout_header=QtGui.QVBoxLayout()
    lb_status=QtGui.QLabel('Label Alerte')
    lb_warning=QtGui.QLabel('Label Warning')
    layout_header.addWidget(lb_status)
    layout_header.addWidget(lb_warning)
    w.setLayout(layout_header)
    return(lb_status,lb_warning)




(w,system)=launch_main_window()
(lb_status,lb_warning)=create_header(w)
sys.exit(system.exec_())