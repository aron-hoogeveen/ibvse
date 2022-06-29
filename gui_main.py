"""
This file contains all the GUI utility
"""
import os
import sys
from PyQt5.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QPlainTextEdit, QApplication,
                             QFileDialog, QFrame, QMessageBox, QTabWidget, QTextBrowser, QComboBox, QSlider, QLabel)
import prototype_main
from PyQt5.QtGui import QIcon
import PyQt5.QtCore as QtCore
from GUI import gui_videoplayer


class InputWindow(QWidget):
    def __init__(self, parent=None):
        """
        Initializer for the class that creates the GUI window
        """
        super().__init__(parent)
        self.setWindowIcon(QIcon(r'GUI/logo.png'))
        self.timeslots = []
        self.layout = QVBoxLayout()
        self.setGeometry(150, 150, 750, 400)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.init_gui()
        self.setLayout(self.layout)
        self.setWindowTitle("BAP group H | Image-Based Video Search Engine")

        self.videos = None  # List containing all the paths to the videos
        self.images = None  # List containing all the paths to the images

    def init_gui(self):
        """
        Build the GUI
        :return: The GUI
        """
        input_layout = QHBoxLayout()  # create the overall layout for the vid/img input block

        # create button to upload videos and text box for feedback
        self.load_vid_btn = QPushButton("Select an input video")
        self.load_vid_btn.clicked.connect(self.vid_in)
        self.load_vid_txt = QPlainTextEdit()
        self.load_vid_txt.setReadOnly(True)

        # Create a layout for the video input
        vid_in_layout = QVBoxLayout()
        vid_in_layout.setContentsMargins(10, 10, 10, 10)
        vid_in_layout.addWidget(self.load_vid_btn)
        vid_in_layout.addWidget(self.load_vid_txt, stretch=1)

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
                                                      "Videos (*.mp4 *.mkv *.mov *.wmv)", options=options)
        self.load_vid_txt.clear()
        print(self.videos)
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
                                                      "Images (*.png *.jpg *.jpeg)", options=options)
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
        if not (self.images and self.videos):
            missing_input = []
            if not self.videos:
                missing_input.append("Video")
            if not self.images:
                missing_input.append("Image")
            self.create_input_error(missing_input)
        else:
            print("==== Started Search =====")
            # Call the function that starts the search engine
            self.setEnabled(False)
            self.repaint()
            self.start_button.setText("Performing search...")

            self.start_button.repaint()
            self.load_img_btn.repaint()
            self.load_vid_btn.repaint()

            results = prototype_main.main(False, self.videos, self.images)
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
        input_error.setWindowIcon(QIcon(r'GUI/logo.png'))
        input_error.setIcon(QMessageBox.Information)
        input_error.setText(f"A problem occurred on starting the search.")
        informative_text = f"Please make sure that the the following inputs are provided:\n\n"
        for missing_input in missing_inputs:
            informative_text += f"- {missing_input}\n"
        input_error.setInformativeText(informative_text)
        input_error.setWindowTitle("BAP group H | Image-Based Video Search Engine")
        input_error.setStandardButtons(QMessageBox.Close)
        input_error.exec_()

    def restore(self):
        """
        Resotre the input window to its original state
        :return:
        """
        self.show()
        self.setEnabled(True)
        self.start_button.setText("Start the Search")


class OutputWindow(QWidget):
    def __init__(self, output_data, videos, images, mainwindow, parent=None):
        """
        Initializer for the class that creates the GUI window
        """
        super().__init__(parent)
        self.setWindowIcon(QIcon(r'GUI/logo.png'))
        self.setWindowTitle("BAP group H | Image-Based Video Search Engine")
        self.cutoffvalue = 113
        self.timeslots = []
        self.output_data = output_data
        self.videos = videos
        self.images = images
        self.mainwindow = mainwindow
        self.layout = QVBoxLayout()
        self.setGeometry(150, 150, 500, 600)
        self.layout.setContentsMargins(10, 10, 10, 10)
        # tabs object to add a tab per video
        self.tab = QTabWidget()
        # button for  a new search
        self.returnBtn = QPushButton("New Search")
        self.returnBtn.clicked.connect(self.return_to_main)
        # slider for the accuracy/amount of results tradeoff
        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(300)
        self.slider.setValue(self.cutoffvalue)
        self.slider.sliderReleased.connect(self.init_gui)
        self.sliderinfo = QLabel("Filter strength \n<= More accurate\t\t\tMore results =>")
        self.sliderinfo.setAlignment(QtCore.Qt.AlignCenter)

        # neat little divider
        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setFrameShadow(QFrame.Sunken)

        # add everything to the layout
        self.layout.addWidget(self.tab)
        self.layout.addWidget(self.sliderinfo)
        self.layout.addWidget(self.slider)
        self.layout.addWidget(divider)
        self.layout.addWidget(self.returnBtn)
        self.setLayout(self.layout)

        # create the videoplayer
        self.videoplayer = gui_videoplayer.VideoPlayer()
        self.init_gui()

    def init_gui(self):
        """
        Build the GUI
        :return: The GUI
        """
        self.cutoffvalue = self.slider.value()
        self.fill_reports()

    def fill_reports(self):
        """
        Creates tabs with content
        :return:
        """
        tabs_to_add = len(self.videos)  # number of tabs = number of videos
        if self.tab.count() != 0:
            self.tab.clear()
        for i in range(tabs_to_add):  # for each video create tab
            tabtoadd = self.create_tabs(self.tab, i, self.videos[i], self.images, self.videoplayer)
            for idx, image in enumerate(self.images):  # for each query image list the occurrences
                timestamps = []
                tabtoadd.tb.append(f"<font size='+2'><b>Input image: {os.path.split(image)[-1]}<br></font></b>")
                matches = False
                for timestamp, dist in zip(self.output_data[i][idx][0], self.output_data[i][idx][1]):
                    if dist > (self.cutoffvalue / 100):  # filter the results
                        break
                    matches = True
                    timestamps.append(timestamp)
                    # add the result to the textbox
                    tabtoadd.tb.append(f"      Occurence at:  "
                                       f"{str(int(round(timestamp // 3600))).zfill(2)}:"
                                       f"{str(int(round((timestamp % 3600) // 60))).zfill(2)}:"
                                       f"{str(int(round((timestamp % 3600) % 60))).zfill(2)}")
                if not matches:
                    tabtoadd.tb.append(f"      No occurrences were found\n")
                else:  # If there are occurences add them to the image/timestamp selector for the videoplayer
                    tabtoadd.cb_image.addItem(os.path.split(image)[-1], timestamps)
                    tabtoadd.update_cb_timestamp(tabtoadd.cb_image.currentIndex())
                tabtoadd.tb.append("\n")

    def create_tabs(self, obj, num, video, images, videoplayer):
        """
        Function to create the tabs
        :param obj: The QTabWidget object
        :param num: The tab number
        :param video: The list of videos
        :param images: The list of images
        :param videoplayer: Videoplayer object reference
        :return: The tab with default layout/contents
        """
        tab_to_add = PageWidget(num, video, images, videoplayer)
        obj.addTab(tab_to_add, os.path.split(self.videos[num])[-1])
        return tab_to_add

    def return_to_main(self):
        """
        Function to go back to the input window
        :return:
        """
        self.hide()
        self.mainwindow.restore()


class PageWidget(QWidget):
    def __init__(self, num, video, images, videoplayer, parent=None):
        super().__init__(parent)
        # set parameters
        self.videoplayer = videoplayer
        self.video = video
        self.images = images

        # Create textbox
        self.tb = QTextBrowser(self)
        self.tb.setFrameShape(QFrame.Box)
        self.tb.setGeometry(QtCore.QRect(10, 10, 400, 500))
        self.tb.setObjectName(str(num))

        # Create combination of widgets to select and image and timestamp
        self.cb_image = QComboBox()
        self.cb_image.currentIndexChanged.connect(self.update_cb_timestamp)
        self.cb_timestamp = QComboBox()
        self.confirmbtn = QPushButton("Play")
        self.confirmbtn.setEnabled(False)
        self.confirmbtn.clicked.connect(self.play_video)
        # Add the widgets to a layout
        self.sublayout = QHBoxLayout()
        self.sublayout.addWidget(self.cb_image)
        self.sublayout.addWidget(self.cb_timestamp)
        self.sublayout.addWidget(self.confirmbtn)

        # Add everything to the overall layout of the tab
        self.layout = QVBoxLayout()
        self.layout.addLayout(self.sublayout)
        self.layout.addWidget(self.tb)
        self.setLayout(self.layout)

    def update_cb_timestamp(self, index):
        """
        Updates the combo box for the timestamps when a different image is selected
        :param index: the index corresponding to the selected image
        :return: Changes the combobox of the timestamps
        """
        self.cb_timestamp.clear()
        self.confirmbtn.setEnabled(True)
        timestamps = self.cb_image.itemData(index)
        for i in range(len(timestamps)):
            timestamps[i] = str(timestamps[i])
        if timestamps:
            self.cb_timestamp.addItems(timestamps)

    def play_video(self):
        """
        Opens the videoplayer at the correct timestamp
        :return: a new window with the videoplayer
        """
        # get the videofile and timestamp from the combobox
        file, timestamp = os.path.abspath(self.video), float(self.cb_timestamp.currentText()) * 1000
        image_path = None
        # Get the correct image path by comparing the file name from the filepath with the name in the combobox
        for image in self.images:
            image_name = os.path.split(image)[-1]
            if image_name == self.cb_image.currentText():
                image_path = image
                break

        assert image_path is not None  # Protection against invalid image path

        if not self.videoplayer.filepath: # if there is no video open a new video
            self.videoplayer.newVid(file, timestamp)
        elif self.videoplayer.filepath == file: # else check if the video was already opened, if so update timestamp
            self.videoplayer.newTimestamp(timestamp)
        else:  # else open the new video
            self.videoplayer.newVid(file, timestamp)

        # displays the correct image
        if not self.videoplayer.image:
            self.videoplayer.new_image(image_path)
        elif image_path != self.videoplayer.image:
            self.videoplayer.new_image(image_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    ex = InputWindow()
    ex.show()
    sys.exit(app.exec_())
