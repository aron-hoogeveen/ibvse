"""
GUI code for the videoplayer
If no video is playing make sure the correct codecs are installed. For Windows:
https://codecguide.com/download_k-lite_codec_pack_basic.htm
"""
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QSlider, QStyle, QVBoxLayout
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton
from PyQt5.QtGui import QIcon, QCloseEvent, QPixmap
import os


class VideoPlayer(QMainWindow):
    def __init__(self):
        """
        Initializes the videplayer window
        """
        super().__init__()
        self.setWindowIcon(QIcon(r'logo.png'))
        self.setWindowTitle("BAP group H | Image-Based Video Search Engine")
        # set params
        self.filepath = None
        self.timestamp = None
        self.image = None
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.setGeometry(150, 150, 700, 400)
        # Create video widget
        videoWidget = QVideoWidget()
        # Create play/pause butotn and slider
        self.playButton = QPushButton()
        self.playButton.setEnabled(False)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.positionSlider)

        # add an image box for the query image
        self.im = QPixmap(os.path.abspath("logo.png"))
        self.im = self.im.scaled(200,200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label = QLabel()
        self.label.setPixmap(self.im)

        # add everything to the layout
        overalllayout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(videoWidget)
        layout.addLayout(controlLayout)
        overalllayout.addLayout(layout)
        overalllayout.addWidget(self.label)

        # Set widget to contain window contents
        wid.setLayout(overalllayout)

        # set mediaplayer paramaters
        self.mediaPlayer.setVideoOutput(videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

    def openFile(self):
        """
        Opens the requested video
        :return:
        """
        if self.filepath != '':
            self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(self.filepath)))
            self.playButton.setEnabled(True)

    def closeEvent(self, a0: QCloseEvent) -> None:
        """
        Resets params after closing the window
        :param a0: Close event trigger
        :return:
        """
        self.filepath = None
        self.timestamp = None
        self.image = None
        self.mediaPlayer.pause()

    def play(self):
        """
        Toggle pause/play of the video
        :return:
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self):
        """
        Update the playbutton icon when the video is (un)paused
        :return:
        """
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self.playButton.setIcon(
                self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        """
        Change the value of the slider
        :param position: The new timestamp (the value for the slider)
        """
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        """
        Changes the range of the slider if the video length is changed
        :param duration: The new length of the video
        """
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        """
        Set the new timestamp (position) of the video
        :param position: the timestamp to change to
        """
        self.mediaPlayer.setPosition(position)


    def newVid(self, filepath, timestamp):
        """
        Opens the videoplayer and loads in a new video
        :param filepath: path to the video
        :param timestamp: the timestamp to open the video on
        :return:
        """
        self.show()
        self.filepath = filepath
        self.timestamp = timestamp
        self.openFile()
        self.mediaPlayer.setPosition(self.timestamp)
        # Play and pause real quick to let the video show
        self.mediaPlayer.play()
        self.mediaPlayer.pause()

    def newTimestamp(self, timestamp):
        """
        Changes the timestamp of an already opened video
        :param timestamp: the timestamp to change to
        """
        self.timestamp = timestamp
        self.mediaPlayer.setPosition(self.timestamp)
        self.mediaPlayer.pause()

    def new_image(self, image):
        """
        Change the query image
        :param image: the path to the image
        """
        self.image = image
        pixmap = QPixmap(self.image)
        pixmap = pixmap.scaled(200,200, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.label.setPixmap(pixmap)

