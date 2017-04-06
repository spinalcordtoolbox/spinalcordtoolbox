import sys
import PyQt4.QtGui as QtGui

a=QtGui.QApplication(sys.argv)
w=QtGui.QWidget()
w.resize(320,240)
w.setWindowTitle('Hello World')
w.show()

sys.exit(a.exec_())