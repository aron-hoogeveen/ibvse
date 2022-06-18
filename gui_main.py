# from PyQt5.QtCore import QDir, Qt, QUrl
# from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
# from PyQt5.QtMultimediaWidgets import QVideoWidget
# from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
#                              QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, qApp)
# from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction
# from PyQt5.QtGui import QIcon, QCloseEvent
# import sys
# import time
# from gui_videplayer import VideoPlayer
#
# class Overview(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("PyQt5 Video Player")
#         self.button1 = QPushButton("SOMETHING", self)
#         self.button2 = QPushButton("SOMETHING ELSE", self)
#         self.button2.move(100,0)
#
#         self.createNewWindow()
#         self.button1.clicked.connect(self.load1)
#         self.button2.clicked.connect(self.load2)
#
#     def createNewWindow(self):
#         self.videoplayerWindow = VideoPlayer()
#         self.videoplayerWindow.resize(640, 480)
#         self.videoplayerWindow.hide()
#
#     def load1(self):
#         filepath, timestamp = r'E:\2022-01-24 20-39-24.mkv', 2000
#         if not self.videoplayerWindow.filepath:
#             self.videoplayerWindow.newVid(filepath, timestamp)
#         else:
#             print('oi')
#
#     def load2(self):
#         filepath, timestamp = r'E:\2022-01-20 13-46-44.mkv', 1000
#         if self.videoplayerWindow.filepath == filepath:
#             self.videoplayerWindow.newTimestamp(timestamp)
#         else:
#             self.videoplayerWindow.newVid(filepath, timestamp)
#
# app = QApplication(sys.argv)
# videoplayer = Overview()
# # videoplayer = VideoPlayer(r'E:\2022-01-24 20-39-24.mkv',2000)
#
# videoplayer.show()
# sys.exit(app.exec_())

import time
import os
import sys
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QPlainTextEdit, QApplication,
                             QFileDialog, QFrame, QMessageBox, QTabWidget,QTextBrowser, QComboBox, QSlider, QLabel)
import prototype_main
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore
import gui_videplayer

class Window(QWidget):
    def __init__(self, parent = None):
        """
        Initializer for the class that creates the GUI window
        """
        super().__init__(parent)
        self.setWindowIcon(QIcon(r'./logo.png'))
        self.timeslots = []
        self.layout = QVBoxLayout()
        self.setGeometry(150, 150, 750, 400)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.init_gui()
        self.setLayout(self.layout)
        self.setWindowTitle("BAP group H | Image Based Video Search Engine")

        self.videos = None  # List containing all the paths to the videos
        self.images = None  # List containing all the paths to the images

    def init_gui(self):
        """
        Build the GUI
        :return: The GUI
        """
        input_layout = QHBoxLayout() # create the overall layout for the vid/img input block

        # create button to upload videos and text box for feedback
        self.load_vid_btn = QPushButton("Select an input video")
        self.load_vid_btn.clicked.connect(self.vid_in)
        self.load_vid_txt = QPlainTextEdit()
        self.load_vid_txt.setReadOnly(True)

        # Create a layout for the video input
        vid_in_layout = QVBoxLayout()
        vid_in_layout.setContentsMargins(10,10,10,10)
        vid_in_layout.addWidget(self.load_vid_btn)
        vid_in_layout.addWidget(self.load_vid_txt, stretch = 1)

        # create button to upload images and text box for feedback
        self.load_img_btn = QPushButton("Select an input image")
        self.load_img_btn.clicked.connect(self.img_in)
        self.load_img_txt = QPlainTextEdit()
        self.load_img_txt.setReadOnly(True)

        # Create a layout for the image input
        img_in_layout = QVBoxLayout()
        img_in_layout.setContentsMargins(10, 10, 10, 10)
        img_in_layout.addWidget(self.load_img_btn)
        img_in_layout.addWidget(self.load_img_txt, stretch=1)

        # Create a nice vertical line as divider
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        # Add all elements to the sub-layout
        input_layout.addLayout(vid_in_layout, stretch=1)
        input_layout.addWidget(divider)
        input_layout.addLayout(img_in_layout, stretch=1)
        # Add to main layout
        self.layout.addLayout(input_layout)

        # Add a button to start the search
        self.start_button = QPushButton("Start the search")
        self.start_button.clicked.connect(self.start_search)
        self.layout.addWidget(self.start_button)

    def vid_in(self):
        """
        Opens the file explorer (build in explorer in pyqt5) and selects the input videos
        :return: A list of input videos and feedback in the text box (only filename)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # Opens multiple files, change to getOpenFileName if you want to force only 1 file
        # To add multiple file formats add them in between the brackets of Videos(*.mp4). Use space to separate them
        self.videos, _ = QFileDialog.getOpenFileNames(self, "Video Select", "",
                                                      "Videos (*.*)", options=options)
        self.load_vid_txt.clear()
        # Add to textbox for feedback
        for video in self.videos:
            video = os.path.split(video)[-1]
            self.load_vid_txt.appendPlainText(video)


    def img_in(self):
        """
        Opens the file explorer (build in explorer in pyqt5) and selects the input images
        :return: A list of input images and feedback in the text box (only filename)
        """
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        # Opens multiple files, change to getOpenFileName if you want to force only 1 file
        # To add multiple file formats add them in between the brackets of Images(*.png). Use space to separate them
        self.images, _ = QFileDialog.getOpenFileNames(self, "Image Select", "",
                                                      "Images (*.*)", options=options)
        self.load_img_txt.clear()
        # Add to textbox for feedback
        for image in self.images:
            image = os.path.split(image)[-1]
            self.load_img_txt.appendPlainText(image)
        print(self.images)

    def start_search(self):
        """
        Check if there are no missing inputs and start the search
        :return: Final result of search
        """
        if not(self.images and self.videos):
            missing_input = []
            if not self.videos:
                missing_input.append("Video")
            if not self.images:
                missing_input.append("Image")
            self.create_input_error(missing_input)
        else:
            print("==== Started Search =====")
            # Call the function that starts the search engine
            results = []  # index, dist
            self.setEnabled(False)
            self.repaint()
            self.start_button.setText("Performing search...")

            self.start_button.repaint()
            self.load_img_btn.repaint()
            self.load_vid_btn.repaint()
            # self.hide()
            for i,video in enumerate(self.videos):
                results.append(prototype_main.main(False, video, self.images))
            self.output_window = OutputWindow(results, self.videos, self.images, self)
            self.output_window.show()
            self.hide()

    def create_input_error(self, missing_inputs):
        """
        Creates an error popup if there are missing inputs
        :param missing_inputs: A list of missing inputs
        :return: An error popup with what is missing
        """

        input_error = QMessageBox()
        input_error.setWindowIcon(QIcon(r'./logo.png'))
        input_error.setIcon(QMessageBox.Information)
        input_error.setText(f"A problem occurred on starting the search.")
        informative_text = f"Please make sure that the the following inputs are provided:\n\n"
        for missing_input in missing_inputs:
            informative_text += f"- {missing_input}\n"
        input_error.setInformativeText(informative_text)
        input_error.setWindowTitle("Missing inputs")
        input_error.setStandardButtons(QMessageBox.Close)
        input_error.exec_()

    def restore(self):
        self.show()
        self.setEnabled(True)
        self.start_button.setText("Start the Search")


class OutputWindow(QWidget):
    def __init__(self, output_data, videos, images, mainwindow, parent = None):
        """
        Initializer for the class that creates the GUI window
        """
        super().__init__(parent)
        self.setWindowIcon(QIcon(r'./logo.png'))
        self.setWindowTitle("BAP group H | Image Based Video Search Engine")
        self.cutoffvalue = 113
        self.timeslots = []
        self.output_data = output_data
        self.videos = videos
        self.images = images
        self.mainwindow = mainwindow
        self.layout = QVBoxLayout()
        self.setGeometry(150, 150, 500, 600)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.tab = QTabWidget()

        self.returnBtn = QPushButton("New Search")
        self.returnBtn.clicked.connect(self.returnToMain)

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(300)
        self.slider.setValue(self.cutoffvalue)
        self.slider.sliderReleased.connect(self.init_gui)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        self.sliderinfo = QLabel("Filter strength \n<= More accurate\t\t\tMore results =>")
        self.sliderinfo.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(self.tab)
        self.layout.addWidget(self.sliderinfo)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(divider)
        self.layout.addWidget(self.returnBtn)


        self.setLayout(self.layout)
        self.setWindowTitle("BAP group H | Video search engine")
        self.videoplayer = gui_videplayer.VideoPlayer()
        print(self.videos)
        self.init_gui()


    def init_gui(self):
        """
        Build the GUI
        :return: The GUI
        """
        print(self.cutoffvalue)
        self.cutoffvalue = self.slider.value()
        print(self.cutoffvalue)
        self.fill_reports()

    def fill_reports(self):
        tabs_to_add = len(self.videos)
        if self.tab.count() != 0:
            self.tab.clear()
        for i in range(tabs_to_add):
            tabtoadd = self.create_tabs(self.tab, i, self.videos[i], self.videoplayer)
            for idx, image in enumerate(self.images):
                timestamps = []
                tabtoadd.tb.append(f"<font size='+2'><b> Query Image: {os.path.split(image)[-1]}<br></font></b>")
                matches = False
                for timestamp, dist in zip(self.output_data[i][idx][0], self.output_data[i][idx][1]):
                    if dist > (self.cutoffvalue/100):
                        break
                    matches = True
                    timestamps.append(timestamp)
                    tabtoadd.tb.append(f"      Occurence at:  "
                                       f"{str(int(round(timestamp//3600))).zfill(2)}:"
                                       f"{str(int(round((timestamp%3600)//60))).zfill(2)}:"
                                       f"{str(int(round((timestamp%3600)%60))).zfill(2)}")
                if not matches:
                    tabtoadd.tb.append(f"No occurences were found")
                else:
                    tabtoadd.cb_image.addItem(os.path.split(image)[-1], timestamps)
                    tabtoadd.update_cb_timestamp(tabtoadd.cb_image.currentIndex())


    def create_tabs(self, obj, num, video, videoplayer):
        tabtoadd = PageWidget(num, video, videoplayer)
        obj.addTab(tabtoadd, os.path.split(self.videos[num])[-1])
        return tabtoadd

    def returnToMain(self):
        self.hide()
        self.mainwindow.restore()




class PageWidget(QWidget):
    def __init__(self, num, video, videoplayer, parent=None):
        super().__init__(parent)

        self.videoplayer = videoplayer
        self.video = video
        self.tb = QTextBrowser(self)
        self.tb.setFrameShape(QFrame.Box)
        self.tb.setGeometry(QtCore.QRect(10, 10, 400, 500))
        self.tb.setObjectName(str(num))

        self.sublayout = QHBoxLayout()
        self.cb_image = QComboBox()
        self.cb_timestamp = QComboBox()
        self.confirmbtn = QPushButton("Play")
        self.confirmbtn.setEnabled(False)
        self.confirmbtn.clicked.connect(self.playvideo)

        self.sublayout.addWidget(self.cb_image)
        self.sublayout.addWidget(self.cb_timestamp)
        self.sublayout.addWidget(self.confirmbtn)

        self.cb_image.currentIndexChanged.connect(self.update_cb_timestamp)

        self.layout = QVBoxLayout()
        self.layout.addLayout(self.sublayout)
        self.layout.addWidget(self.tb)
        self.setLayout(self.layout)

    def update_cb_timestamp(self, index):
        self.cb_timestamp.clear()
        self.confirmbtn.setEnabled(True)
        timestamps = self.cb_image.itemData(index)
        for i in range(len(timestamps)):
            timestamps[i] = str(timestamps[i])
        if timestamps:
            self.cb_timestamp.addItems(timestamps)

    def playvideo(self):
        file, timestamp = os.path.abspath(self.video), float(self.cb_timestamp.currentText())*1000
        if not self.videoplayer.filepath:
            self.videoplayer.newVid(file, timestamp)
        elif self.videoplayer.filepath == file:
            self.videoplayer.newTimestamp(timestamp)
        else:
            self.videoplayer.newVid(file, timestamp)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = Window()
    ex.show()
    sys.exit(app.exec_())