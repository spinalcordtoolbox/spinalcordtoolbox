import sys
import PyQt4.QtGui as QtGui
import PyQt4.QtCore as QtCore


def launch_main_window():
    system = QtGui.QApplication(sys.argv)
    w = QtGui.QWidget()
    w.resize(740, 850)
    w.setWindowTitle('Hello World')
    w.show()
    return (w,system)

def create_header(w):
    layout_header=QtGui.QVBoxLayout()
    layout_header.setAlignment(QtCore.Qt.AlignTop)
    lb_status=QtGui.QLabel('Label Alerte')
    lb_status.setContentsMargins(10, 10, 10, 0)
    lb_status.setAlignment(QtCore.Qt.AlignCenter)

    lb_warning=QtGui.QLabel('Label Warning')
    lb_warning.setContentsMargins(10,10,10,10)
    lb_warning.setAlignment(QtCore.Qt.AlignCenter)
    layout_header.addWidget(lb_status)
    layout_header.addWidget(lb_warning)
    w.setLayout(layout_header)
    return(lb_status,lb_warning)



(w,system)=launch_main_window()
(lb_status,lb_warning)=create_header(w)
sys.exit(system.exec_())