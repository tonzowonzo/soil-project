# Soil GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMessageBox, QDesktopWidget, QMainWindow, \
QAction, qApp, QMenu, QHBoxLayout, QVBoxLayout, QInputDialog, QLineEdit, QFileDialog, QLabel
from PyQt5.QtGui import QIcon, QFont, QPixmap  
from PyQt5.QtCore import QCoreApplication

    
class SoilGui(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        QToolTip.setFont(QFont('SansSerif', 10)) # Size 10 SansSerif font.
        self.setToolTip('This is a <b>QWidget</b> widget') # Shows the tool tip when hovering over the window.
        # Import button
        btn1 = QPushButton('Import', self) # Button object, called import.
        btn1.clicked.connect(self.getFile)
        btn1.resize(btn1.sizeHint()) # Gives a recommended size for the button.
        btn1.move(50, 530) # Moves the button.
        # Export button
        btn2 = QPushButton('Export', self) 
        btn2.resize(btn2.sizeHint()) 
        btn2.move(150, 530) 
        
    
        # Menu bar stuff
        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        # Sub menu
        importMenu = QMenu('Import', self)
        importImageAct = QAction('Import image', self) # Import image submenu
        importDataAct = QAction('Import data', self) # Import data submenu
        importMenu.addActions([importImageAct, importDataAct])
               
        exportDataMenu = QMenu('Export', self)
        exportDataAct = QAction('Export data', self) # Export data submenu
        exportDataMenu.addAction(exportDataAct)
        
        fileMenu.addMenu(importMenu)
        fileMenu.addMenu(exportDataMenu)
                
        self.statusBar().showMessage('Ready') # Sets a status bar which says ready.
        self.setGeometry(900, 600, 900, 600) # Sets the size and place of the window.
        self.center()
        self.setWindowTitle('Soil Classifier') # Sets title of the window.
        self.setWindowIcon(QIcon('soilImg.png')) # Adds an icon to the window.
        self.show() # Shows the window.        
    
    def getFile(self):
        l1 = QLabel(self)
        image = QFileDialog.getOpenFileName(None,'OpenFile','',"Image file(*.jpg *.png)")
        imagePath = image[0]
        pixmap = QPixmap(imagePath)
        l1.setPixmap(pixmap)
        l1.move(50, 50)
        l1.resize(100, 100)
        l1.show()
    
        
    def center(self):
        '''
        Centers the window in the middle of the screen.
        '''
        qr = self.frameGeometry() # Specifies the geometry of the main window.
        cp = QDesktopWidget().availableGeometry().center() # Screen resolution of monitor.
        qr.moveCenter(cp) # Moves the window to the center.
        self.move(qr.topLeft()) # Move the top left of the window to the top left of our centred rectangle
        
        
    def closeEvent(self, event):
        '''
        Causes a message box to pop up when pressing the X button.
        '''
        reply = QMessageBox.question(self, 'Message', 'Are you sure you want to quit?',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        # The last .No is the button the mouse automatically focusses on first.
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

#    def getFile(self):
##        fname = QFileDialog.getOpenFileName(self, 'Open File',
##                                            'c:\\', 'Image files (*.jpg *.png') # Opens browse
##        pixMap = QPixmap(fname)
#        self.l1.setPixmap(QPixmap('lilyDog.jpg'))
if __name__ == '__main__':
    app = QApplication(sys.argv) 
    w = SoilGui() # Opens an instance of the SoilGui class.
    sys.exit(app.exec_()) # Allows a clean exit of the application.
    
