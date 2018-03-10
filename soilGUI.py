# Soil GUI
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QToolTip, QMessageBox, QDesktopWidget, QMainWindow, \
QAction, qApp, QMenu, QHBoxLayout, QVBoxLayout, QInputDialog, QLineEdit, QFileDialog, QLabel, QStyleFactory
from PyQt5.QtGui import QIcon, QFont, QPixmap
from PyQt5.QtCore import QCoreApplication
import PyQt5.QtCore
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
plt.style.use(['ggplot', 'dark_background'])
#Load keras models
from keras.models import load_model
model = load_model('SoilEnhancedPretrained2_12class.h5')
model2 = load_model('SoilEnhancedPretrained12class.h5')
#ANN_model = load_model('ANNsoilUSDA.h5')

# Load random forest model.
from sklearn.externals import joblib
rand_forest_soil = joblib.load('random_forest_soil.pkl')


class SoilGui(QMainWindow):
        
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.isPressed = False
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
        pHlabel = QLabel(self)
        pHlabel.setText('Enter topsoil pH')
        pHlabel.resize(180, 40)
        pHlabel.move(650, 90)
        pHlabel.setToolTip('''
        The pH of the water in the topsoil.
                           ''')
        self.pH = QLineEdit(self) # Text box for pH levels
        self.pH.move(650, 120)
        self.pH.resize(180, 40)
        self.pH.setText('Topsoil pH')
        
        Drainagelabel = QLabel(self)
        Drainagelabel.setText('Enter drainage class')
        Drainagelabel.resize(180, 40)
        Drainagelabel.move(650, 160)
        Drainagelabel.setToolTip('''
        A drainage class is a value between
        1 and 7 where drainage is 1: Very poor, 
        2: Poor, 3: Imperfectly, 4: Moderately well,
        5: Well, 6: Somewhat excessive, 7: Excessive
                                 ''')
        self.drainage = QLineEdit(self) # Text box for soil grain size in mm
        self.drainage.move(650, 190)
        self.drainage.resize(180, 40)
        self.drainage.setText('Drainage class')

        texlabel = QLabel(self)
        texlabel.setText('Enter USDA texture')
        texlabel.resize(180, 40)
        texlabel.move(650, 230)
        texlabel.setToolTip('''
        USDA texture is a value between 1 and 13 where
        1: clay(heavy), 2: silty clay, 3: clay(light),
        4: silty clay loam, 5: clay loam, 6: silt,
        7: silt loam, 8: sandy clay, 9: loam,
        10: sandy clay loam, 11: sandy loam, 12: loamy sand
        and 13: sand
                            ''')
        self.texture = QLineEdit(self) # Text box for pH levels
        self.texture.move(650, 260)
        self.texture.resize(180, 40)
        self.texture.setText('Topsoil USDA texture class')

        toplabel = QLabel(self)
        toplabel.setText('Enter topsoil bulk density')
        toplabel.resize(180, 40)
        toplabel.move(650, 300)
        toplabel.setToolTip('''
        Soil bulk density is a number which generally
        is around the values of 1 to 1.6g/cm^3 however,
        it can vary largely. In this case it is the field
        bulk density (wet).                 
                            ''')
        self.bulkd = QLineEdit(self) # Text box for pH levels
        self.bulkd.move(650, 330)
        self.bulkd.resize(180, 40)
        self.bulkd.setText('Topsoil bulk density')
                          
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
        self.isPressed = False
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
        self.TreePredictLabel.deleteLater()
        
    def predictImage(self):
        '''
        Predicts the image class based on the input from keras CNN model.
        '''
        if self.isPressed == False:
            self.isPressed = True
            self.predictLabel = QLabel(self)
            self.predictLabel.move(50, 430)
            self.predictLabel.resize(250, 100)
            self.predictLabel.show()
            img = image.load_img(self.imagePath, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255
            
            # Ensemble with 2 models
            pred1 = model.predict(x)
            pred2 = model2.predict(x)
            pred = (pred1 + pred2) / 2
            print(pred)
            
            # Plot the probabilities of each soil type
            plt.barh([i for i in range(12)], pred[0])
            plt.grid('off')
            plt.xlabel('Probability')
            plt.ylabel('Soil class')
            plt.show()
            
    #        indexes = np.argpartition(pred, 10)[-2:]
            indexes = np.argsort(pred)[-2:]
            print(indexes)
            # Predict multi class classification
            if indexes[0][11] == 0:
                soil1 = 'Alfisol'
                probability1 = pred[0][0] * 100
            elif indexes[0][11] == 1:
                soil1 = 'Andisol'
                probability1 = pred[0][1] * 100
            elif indexes[0][11] == 2:
                soil1 = 'Aridisol'
                probability1 = pred[0][2] * 100
            elif indexes[0][11] == 3:
                soil1 = 'Entisol'
                probability1 = pred[0][3] * 100
            elif indexes[0][11] == 4:
                soil1 = 'Histosol'
                probability1 = pred[0][4] * 100
            elif indexes[0][11] == 5:
                soil1 = 'Inceptisol'
                probability1 = pred[0][5] * 100
            elif indexes[0][11] == 6:
                soil1 = 'Mollisol'
                probability1 = pred[0][6] * 100
            elif indexes[0][11] == 7:
                soil1 = 'Non soil'
                probability1 = pred[0][7] * 100
            elif indexes[0][11] == 8:
                soil1 = 'Oxisol'
                probability1 = pred[0][8] * 100
            elif indexes[0][11] == 9:
                soil1 = 'Spodosol'
                probability1 = pred[0][9] * 100
            elif indexes[0][11] == 10:
                soil1 = 'Ultisol'
                probability1 = pred[0][10] * 100
            elif indexes[0][11] == 11:
                soil1 = 'Vertisol'
                probability1 = pred[0][11] * 100
            print(soil1, probability1)
            if indexes[0][10] == 0:
                soil2 = 'Alfisol'
                probability2 = pred[0][0] * 100
            elif indexes[0][10] == 1:
                soil2 = 'Andisol'
                probability2 = pred[0][1] * 100
            elif indexes[0][10] == 2:
                soil2 = 'Aridisol'
                probability2 = pred[0][2] * 100
            elif indexes[0][10] == 3:
                soil2 = 'Entisol'
                probability2 = pred[0][3] * 100
            elif indexes[0][10] == 4:
                soil2 = 'Histosol'
                probability2 = pred[0][4] * 100
            elif indexes[0][10] == 5:
                soil2 = 'Inceptisol'
                probability2 = pred[0][5] * 100
            elif indexes[0][10] == 6:
                soil2 = 'Mollisol'
                probability2 = pred[0][6] * 100
            elif indexes[0][10] == 7:
                soil2 = 'Non soil'
                probability2 = pred[0][7] * 100
            elif indexes[0][10] == 8:
                soil2 = 'Oxisol'
                probability2 = pred[0][8] * 100
            elif indexes[0][10] == 9:
                soil2 = 'Spodosol'
                probability2 = pred[0][9] * 100
            elif indexes[0][10] == 10:
                soil2 = 'Ultisol'
                probability2 = pred[0][10] * 100
            elif indexes[0][10] == 11:
                soil2 = 'Vertisol'
                probability2 = pred[0][11] * 100
            print(soil2, probability2)
    
            self.predictLabel.setText('This is a {} with {}% probability \nOr a {} with {}% probability'.format(soil1,
                                      str(round(probability1, 2)), soil2, str(round(probability2, 2))))
            print('This is a {} with {}% probability \n Or a {} with {}% probability'.format(soil1,
                                      str(round(probability1, 2)), soil2, str(round(probability2, 2))))
            pH = None
            bulk_density = None
            drainage_class = None
            USDA_texture_class = None
            try:
                if 0 < float(self.pH.text()) < 15:
                    pH = float(self.pH.text())
                    print(pH)                    
            except:
                pass        
            try:
                if 0 < float(self.bulkd.text()) < 4:
                    bulk_density = float(self.bulkd.text())
                    print(bulk_density)
            except ValueError:
                pass            
            if self.drainage.text().isdigit():
                drainage_class = int(self.drainage.text())
                if drainage_class < 8 and drainage_class > 0:
                    print(drainage_class)
            if self.texture.text().isdigit() and 0 < int(self.texture.text()) < 14:
                USDA_texture_class = int(self.drainage.text())
                print(USDA_texture_class)
                
            if None in [pH, bulk_density, drainage_class, USDA_texture_class]:
                print('You did not input a or several numbers')
            else:
                nums_for_prediction = np.array([drainage_class, USDA_texture_class, bulk_density, pH])
                nums_for_prediction = nums_for_prediction.reshape(1, -1)
                prediction_forest = rand_forest_soil.predict(nums_for_prediction)
                soil_type = np.argmax(prediction_forest)
                print(prediction_forest)
                
                if soil_type == 0:
                    soil = 'Random forest classifier: This is an Alfisol'
                if soil_type == 1:
                    soil = 'Random forest classifier: This is an Andisol'
                elif soil_type == 2:
                    soil = 'Random forest classifier: This is an Aridisol'
                elif soil_type == 3:
                    soil = 'Random forest classifier: This is an Entisol'
                elif soil_type == 4:
                    soil = 'Random forest classifier: This is a Histosol'
                elif soil_type == 5:
                    soil = 'Random forest classifier: This is an Inceptisol'
                elif soil_type == 6:
                    soil = 'Random forest classifier: This is a Mollisol'
                elif soil_type == 7:
                    soil = 'Random forest classifier: This is an Oxisol'
                elif soil_type == 8:
                    soil = 'Random forest classifier: This is a Spodosol'
                elif soil_type == 9:
                    soil ='Random forest classifier: This is an Ultisol'
                elif soil_type == 10:
                    soil = 'Random forest classifier: This is a Vertisol'
                    
                self.TreePredictLabel = QLabel(self)
                self.TreePredictLabel.move(580, 330)
                self.TreePredictLabel.resize(300, 100)
                self.TreePredictLabel.setText(soil)
                self.TreePredictLabel.show()
            


            
    def center(self):
        '''
        Centers the window in the middle of the screen.
        '''
        qr = self.frameGeometry() # Specifies the geometry of the main window.
        cp = QDesktopWidget().availableGeometry().center() # Screen resolution of monitor.
        qr.moveCenter(cp) # Moves the window to the center.
        self.move(qr.topLeft()) # Move the top left of the window to the top left of our centred rectangle
        
if __name__ == '__main__':
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        app.setStyle(QStyleFactory.create('Windows'))
    w = SoilGui() # Opens an instance of the SoilGui class.
    sys.exit(app.exec_()) # Allows a clean exit of the application.

