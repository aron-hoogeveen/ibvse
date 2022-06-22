from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, qApp)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QCloseEvent, QPixmap
import sys
import time
import os


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowIcon(QIcon(r'./logo.png'))
        self.setWindowTitle("BAP group H | Image-Based Video Search Engine")
        self.filepath = None
        self.timestamp = None
        self.image = None
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()
        self.setGeometry(150, 150, 700, 400)


        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.error = QLabel()
        self.error.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        self.im = QPixmap(os.path.abspath("logo.png"))
        self.im = self.im.scaled(200,200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label = QLabel()
        self.label.setPixmap(self.im)

        overalllayout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)

        overalllayout.addLayout(layout)
        overalllayout.addWidget(self.label)

        # Set widget to contain window contents
        wid.setLayout(overalllayout)

        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)


    def openFile(self):
        if self.filepath != '':
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(self.filepath)))
            self.playButton.setEnabled(True)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.filepath = None
        self.timestamp = None
        self.image = None
        self.mediaPlayer.pause()
        # self.mediaPlayer.pause()
        # self.mediaPlayer.deleteLater()
        # self.close()

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        print(self.mediaPlayer.error)
        self.playButton.setEnabled(False)
        self.error.setText("Error: " + self.mediaPlayer.errorString())

    def newVid(self, filepath, timestamp):
        self.show()
        self.filepath = filepath
        self.timestamp = timestamp
        self.openFile()
        self.mediaPlayer.setPosition(self.timestamp)
        self.mediaPlayer.play()
        self.mediaPlayer.pause()
    def newTimestamp(self, timestamp):
        self.timestamp = timestamp
        self.mediaPlayer.setPosition(self.timestamp)
        self.mediaPlayer.pause()

    def new_image(self, image):
        self.image = image
        print(image)
        print(self.image)
        pixmap = QPixmap(self.image)
        pixmap = pixmap.scaled(200,200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label.setPixmap(pixmap)

