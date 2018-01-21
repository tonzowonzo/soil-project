# Soil GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMessageBox, QDesktopWidget, QMainWindow, \
QAction, qApp, QMenu, QHBoxLayout, QVBoxLayout, QInputDialog, QLineEdit, QFileDialog, QLabel, QStyleFactory
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QCoreApplication
import PyQt5.QtCore
import numpy as np
from keras.preprocessing import image
#Load keras model
from keras.models import load_model
model = load_model('soilNetPretrained4class.h5')
    
class SoilGui(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        # Import button
        btn1 = QPushButton('Import', self) # Button object, called import.
        btn1.clicked.connect(self.getFile)
        self.image_matrix = self.getFile
        btn1.resize(btn1.sizeHint()) # Gives a recommended size for the button.
        btn1.move(50, 530) # Moves the button.
        # Export button
        btn2 = QPushButton('Export', self) 
        btn2.resize(btn2.sizeHint()) 
        btn2.move(150, 530) 
        # Analyse button
        btn3 = QPushButton('Analyse', self) 
        btn3.clicked.connect(self.predictImage)
        btn3.resize(btn2.sizeHint()) 
        btn3.move(250, 530) 
        # Text boxes
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 50)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.gSize = QLineEdit(self) # Text box for soil grain size in mm
        self.gSize.move(700, 100)
        self.gSize.resize(180, 40)
        self.gSize.setText('Enter your soils grain size')

        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 150)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 200)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 250)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 300)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 350)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(700, 400)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 50)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.gSize = QLineEdit(self) # Text box for soil grain size in mm
        self.gSize.move(480, 100)
        self.gSize.resize(180, 40)
        self.gSize.setText('Enter your soils grain size')

        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 150)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 200)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 250)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 300)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 350)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
        
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(480, 400)
        self.pH.resize(180, 40)
        self.pH.setText('Enter your soils pH')
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
    
    def getFile(self, image_matrix):
        '''
        Fetches an image file from a file menu and display
        '''
        l1 = QLabel(self)
        image = QFileDialog.getOpenFileName(None,'OpenFile','',"Image file(*.jpg *.png)")
        self.imagePath = image[0]
        pixmap = QPixmap(self.imagePath)
        pixmap = pixmap.scaled(299, 299)
        l1.setPixmap(pixmap)
        l1.move(50, 50)
        l1.resize(299, 299)
        l1.show()
        self.predictLabel.deleteLater()
        
    def predictImage(self):
        '''
        Predicts the image class based on the input from keras CNN model.
        '''
        self.predictLabel = None
        self.predictLabel = QLabel(self)
        self.predictLabel.move(50, 500)
        self.predictLabel.resize(250, 40)
        self.predictLabel.show()
        img = image.load_img(self.imagePath, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255
        pred = model.predict(x)
        print(pred)
        className = pred.argmax()
        print(className)
        # Predict multi class classification
        if className == 0:
            soil = 'Mollisol'
            probability = pred[0][0] * 100
        elif className == 1:
            soil = 'Oxisol'
            probability = pred[0][1] * 100
        elif className == 2:
            soil = 'Spodosol'
            probability = pred[0][2] * 100
        elif className == 3:
            soil = 'Vertisol'
            probability = pred[0][3] * 100
        self.predictLabel.setText('This is a {} with {} probability'.format(soil, str(round(probability, 2))))
 
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

if __name__ == '__main__':
    app = QApplication(sys.argv) 
    app.setStyle(QStyleFactory.create('Windows'))
    w = SoilGui() # Opens an instance of the SoilGui class.
    sys.exit(app.exec_()) # Allows a clean exit of the application.

