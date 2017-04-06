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
    header_layout=QtGui.QVBoxLayout()
    lb_status=QtGui.QLabel('Hi there!')
    header_layout.addWidget(lb_status)
    
    w.setLayout(header_layout)




(w,system)=launch_main_window()
create_header(w)
sys.exit(system.exec_())