import sys
import cv2
import numpy as np
import os
import subprocess
from os import path
from collections import Counter
from collections import deque
import keras
import time
import threading
from PyQt5.QtWidgets import QApplication,QDesktopWidget, QWidget,QLabel,QTableWidget,QTableWidgetItem, QPushButton,QFileDialog, QHBoxLayout, QGroupBox, QDialog, QVBoxLayout, QGridLayout

from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.figure import Figure
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import imutils

from PyQt5.QtCore import *
from PyQt5.QtGui import *

class Model():
    def resource_path(self):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        return base_path
    def __init__(self):
        print('Creating model...')
        self.modelJson = 'Model_loss_ES_40Ep.json'
        self.modelH5 = 'Model_loss_ES_40Ep.h5'
        pfrClassesPath = 'pfrClasses.npy'   
        ftClassesPath = 'ftClasses.npy'
        
        
        pfrClassesPath = os.path.join(self.resource_path(),pfrClassesPath)
        ftClassesPath = os.path.join(self.resource_path(),ftClassesPath)
        
        self.pfrEncoder = LabelEncoder()
        self.pfrEncoder.classes_ = np.load(pfrClassesPath,allow_pickle=True)
        self.ftEncoder = LabelEncoder()
        self.ftEncoder.classes_ = np.load(ftClassesPath,allow_pickle=True)
        
        print('Loading Model')
        self.model = self.loadModel()
        print('Creating model...Done')

    def labelDecoder(self,label,cls):
        encoder = None
        if cls == 'PFR':
            encoder = self.pfrEncoder
        elif cls == 'FT':
            encoder = self.ftEncoder
        trueLab = encoder.inverse_transform(label)
        return(trueLab)
    
    def loadModel(self):
            
        modelJsonPath = self.modelJson
        modelH5Path = self.modelH5
        
        modelJsonPath = os.path.join(self.resource_path(),modelJsonPath)
        modelH5Path = os.path.join(self.resource_path(),modelH5Path)
        
        print(modelJsonPath+'##################'+modelH5Path)
        
        with open(modelJsonPath, 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(modelH5Path)
        return model
    
    def classify(self,image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        image = image.astype("float32")/255
        #mean = np.array([123.68, 116.779, 103.939][::-1], dtype="float32")
        #image -= mean
        
        # pass the image through the network to obtain our prediction
        pfr,ft = self.model.predict(np.expand_dims(image, axis=0))
        pfr,ft = pfr.argmax(axis=-1), ft.argmax(axis=-1)
        truePFR,trueFT = self.labelDecoder(pfr,'PFR'),self.labelDecoder(ft,'FT')
        return truePFR,trueFT
    
class Thread(QThread):
    
    infLabel = pyqtSignal(str)
    changePixmap = pyqtSignal(QImage)
    trgLabel = pyqtSignal(str)
    def __init__(self,appData):

        QThread.__init__(self,appData)
        
        print('Thread init')
        
        if appData.model == None:
            self.model = Model()
        else:
            self.model = appData.model
        self.file = appData.file
        
        self.rollAveragePFR = deque([])
        self.rollAverageFT = deque([])
        print('Thread init Done')
        
    
    def rollAverage(self,pfr,ft):
        #handle pfr rolling average
        if len(self.rollAveragePFR) == 10:
            self.rollAveragePFR.rotate(-1)
            self.rollAveragePFR.pop()
            self.rollAveragePFR.append(pfr[0])
        else:
            self.rollAveragePFR.append(pfr[0])
    
        if len(self.rollAverageFT) == 10:
            self.rollAverageFT.rotate(-1)
            self.rollAverageFT.pop()
            self.rollAverageFT.append(ft[0])
        else:
            self.rollAverageFT.append(ft[0])
        
        pfrVals = list(Counter(self.rollAveragePFR).keys())
        pfrValsCounts = list(Counter(self.rollAveragePFR).values())
        maxInd = np.argmax(pfrValsCounts)
        pfr = pfrVals[maxInd]
        
        ftVals = list(Counter(self.rollAverageFT).keys())
        ftValsCounts = list(Counter(self.rollAverageFT).values())
        maxInd = np.argmax(ftValsCounts)
        ft = ftVals[maxInd]
        
        return pfr,ft
        
    def run(self):  
        print('running thread')
        self.infLabel.emit('Video loaded. Analyzing...')
        vidcap = cv2.VideoCapture(self.file)
        success,image = vidcap.read()
       
        while success:
            output = image.copy()
            output = imutils.resize(output, width=400)
            
            truePFR,trueFT = self.model.classify(image)
            print(truePFR,trueFT)
            truePFR,trueFT = self.rollAverage(truePFR,trueFT)
            
            
            pfrtext = "PFR : {pfr}".format(pfr =truePFR)
            fttext =  "Fuel Type : {ft}".format(ft=trueFT)
            
            cv2.putText(output, pfrtext, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            cv2.putText(output, fttext, (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
            #h, w, ch = output.shape
            #bytesPerLine = ch * w
            #p = QImage(image, w, h, bytesPerLine, QImage.Format_RGB888)
            output = QtGui.QImage(output.data, output.shape[1], output.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
            #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(output)
            self.trgLabel.emit(pfrtext+'\n'+fttext)
            self.infLabel.emit('Running pedictions on each frame ...')
            success,image = vidcap.read()
        print('Done')
        self.infLabel.emit('Finished! Click "Load File" to analyze again.')
        
        
class ThreadImage(QThread):
    changePixmapImage = pyqtSignal(QImage)
    infLabel = pyqtSignal(str)
    trgLabel = pyqtSignal(str)
    def __init__(self,appData):
        self.file = appData.file
        
        QThread.__init__(self,appData)
        print('Thread Image init')
        
        if appData.model == None:
            self.model = Model()
        else:
            self.model = appData.model
        self.file = appData.file
            
        print('Thread init Done')
        
    def run(self):  
        print('running thread Image')
        print('Analyzing data..')
        self.infLabel.emit('Image loaded. Analyzing...')
        try:
            img = cv2.imread(self.file)
            output = img.copy()
            output = imutils.resize(output, width=400)
        except Exception as e:
            print('Error in reading Image: ',e)
             
        truePFR,trueFT = self.model.classify(img)
        
         # Image write
        pfrtext = "PFR : {pfr}".format(pfr =truePFR)
        fttext =  "Fuel Type : {ft}".format(ft=trueFT)
        
        cv2.putText(output, pfrtext, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.putText(output, fttext, (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        output = QtGui.QImage(output.data, output.shape[1],output.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.trgLabel.emit(pfrtext+'\n'+fttext)
        self.changePixmapImage.emit(output)
        self.infLabel.emit('Finished! Click "Load File" to analyze again.')
        
        
class App(QWidget):

    def resource_path(self,relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        except Exception:
            base_path = os.path.abspath(".")
        
        return os.path.join(base_path, relative_path)
    def __init__(self):
        super().__init__()
        self.title = 'Flame Characterization -  Siemens Turbomachinery AB'
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 100
        self.model = None
        self.output = None
        self.file = None
        self.initUI()
        
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
        
    @pyqtSlot(str)
    def setLabel(self, text):
        self.infoLabel.setText(text)
        
    @pyqtSlot(str)
    def setTargetLabel(self, text):
        self.targetLabel.setText(text)
        
    def fileExists(self,file):
        if os.path.exists(file):
            return True
        else:
            return False
    
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setFixedSize(900,800)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        self.window()
        
        app = QApplication(sys.argv)
        win = QWidget() 
        win.setFixedSize(900,800)

        self.toolLabel = QLabel()
        self.infoLabel = QLabel()
        self.label = QLabel()
        self.contactLabel = QtWidgets.QPushButton('Contact', self)
        self.contactLabel.setFixedSize(100,20)
        
        self.toolLabel.setText("Flame Characterization Tool")
        self.toolLabel.setStyleSheet('font-size:40px')
        self.contactLabel.setText("Contact")
        self.infoLabel.setText('No file loaded! Please click "Load File" to load file')
        self.infoLabel.setStyleSheet('color:red;font-size:20px')
        
        self.targetLabel = QLabel()
        self.targetLabel.setText('')
        self.targetLabel.setStyleSheet('color:blue;font-size:15px')
        
        self.browseBtn = QtWidgets.QPushButton('Load File', self)
        self.browseBtn.setMaximumWidth(100)
        self.browseBtn.clicked.connect(self.getfiles)
    
        self.toolLabel.setAlignment(Qt.AlignCenter)
        self.infoLabel.setAlignment(Qt.AlignLeft)
        self.label.setAlignment(Qt.AlignCenter)
        
        imPath= 'Siemens.jpg'
        
        imPath = self.resource_path(imPath)
        print('logo: ',imPath)
        
        pixmap = QPixmap(imPath)
        self.label.setPixmap(pixmap)
        self.label.setFixedSize(480,480)
        
        self.contactLabel.clicked.connect(self.cWindow)
        
        
        
        vbox = QVBoxLayout()
        vbox.addWidget(self.toolLabel)
        vbox.addStretch()
        vbox.addWidget(self.infoLabel)
        vbox.addStretch()
        vbox.addWidget(self.browseBtn)
        vbox.addStretch()
        vbox.addWidget(self.targetLabel)
        vbox.addStretch()
        vbox.addWidget(self.label)
        vbox.addStretch()
        vbox.addWidget(self.contactLabel,alignment=Qt.AlignRight)


        win.setLayout(vbox)
    
        win.setWindowTitle("FC Demo")
        win.show()
        sys.exit(app.exec_())
        
    def cWindow(self):
      self.contactInf = QLabel('Info',self)
      self.contactInf.setText('This tool is developed as a part of Master\'s Thesis titled as "Multi Task Convolutional Learning for flame characterization", \ndevloped by: Obaid Ur Rehman')
      self.clabel = QLabel("Contact", self)
      self.clabel.setText('Contact:\nName:Obaid Ur Rehman\nEmail:obaidurrehman1994@gmail.com\nPhone:+46761593548\nLinkedIn:https://www.linkedin.com/in/obaidurrehman1994')
      self.clabel.move(0,50)
      self.setWindowTitle(self.title)
      self.setFixedSize(700,150)
      self.setGeometry(self.top, self.left, self.width, self.height)
      centerPoint = QDesktopWidget().availableGeometry().center()
      self.move(centerPoint)
      self.show()
    def getfiles(self):
      dlg = QFileDialog()
      dlg.setFileMode(QFileDialog.AnyFile)
      dlg.setNameFilters(["Images (*.jpeg  *.png *.jpg)","Videos (*.mp4 *.avi)"])
      
      filenames = []
    		
      if dlg.exec_():
         filenames = dlg.selectedFiles()
         file = filenames[0]
         if self.fileExists(file):
            if '.jpg' in file or '.jpeg' in file or '.png' in file:
             print('Image file found: ',file)
             try:
                 pixmap = QPixmap(file)
                 self.label.setPixmap(pixmap)
                 self.label.resize(480, 480)
                 self.file = file
                 

                 th = ThreadImage(self)
                 th.infLabel.connect(self.setLabel)
                 th.trgLabel.connect(self.setTargetLabel)
                 th.changePixmapImage.connect(self.setImage)
                 
                 th.start() 
               
                 #self.show()
                 print('th killed')
                 
             except Exception as e:
                 print(e)
            else:
             self.file = file
             self.infoLabel.setText('Video Loaded. Please wait, analyzing...')
             
             th = Thread(self)
             th.infLabel.connect(self.setLabel)
             th.trgLabel.connect(self.setTargetLabel)
             th.changePixmap.connect(self.setImage)
             th.start() 
             print('th killed')
             self.infoLabel.setText('Finished!')
         else:
            pass
    
    def analyse(self):
        print('Analyzing data..')
        try:
            img = cv2.imread(self.file)
            self.output = img.copy()
            self.output = imutils.resize(self.output, width=400)
        except Exception as e:
            print('Error in reading Image: ',e)
             
        truePFR,trueFT = self.model.classify(img)
        
         # Image write
        pfrtext = "PFR : {pfr}".format(pfr =truePFR)
        fttext =  "Fuel Type : {ft}".format(ft=trueFT)
        cv2.putText(self.output, pfrtext, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        cv2.putText(self.output, fttext, (3, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0, 255, 0), 2)
        self.output = QtGui.QImage(self.output.data, self.output.shape[1], self.output.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QtGui.QPixmap.fromImage(self.output))
     
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())