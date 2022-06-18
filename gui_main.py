from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                             QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, qApp)
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QCloseEvent
import sys
import time


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Player")
        self.filepath = None
        self.timestamp = None
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        videoWidget = QVideoWidget()

        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
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

        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        layout.addWidget(self.error)

        # Set widget to contain window contents
        wid.setLayout(layout)


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

    def newTimestamp(self, timestamp):
        self.timestamp = timestamp
        self.mediaPlayer.setPosition(self.timestamp)


class Overview(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 Video Player")
        self.button1 = QPushButton("SOMETHING", self)
        self.button2 = QPushButton("SOMETHING ELSE", self)
        self.button2.move(100,0)

        self.createNewWindow()
        self.button1.clicked.connect(self.load1)
        self.button2.clicked.connect(self.load2)

    def createNewWindow(self):
        #
        self.videoplayerWindow = VideoPlayer()
        self.videoplayerWindow.resize(640, 480)
        self.videoplayerWindow.hide()

    def load1(self):
        filepath, timestamp = r'E:\2022-01-24 20-39-24.mkv', 2000
        if not self.videoplayerWindow.filepath:
            self.videoplayerWindow.newVid(filepath, timestamp)
        else:
            print('oi')

    def load2(self):
        filepath, timestamp = r'E:\2022-01-20 13-46-44.mkv', 1000
        if self.videoplayerWindow.filepath == filepath:
            self.videoplayerWindow.newTimestamp(timestamp)
        else:
            self.videoplayerWindow.newVid(filepath, timestamp)

app = QApplication(sys.argv)
videoplayer = Overview()
# videoplayer = VideoPlayer(r'E:\2022-01-24 20-39-24.mkv',2000)

videoplayer.show()
sys.exit(app.exec_())