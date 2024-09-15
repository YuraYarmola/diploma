import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QSlider, QHBoxLayout, QVBoxLayout, QWidget, \
    QFileDialog, QRadioButton
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2


class Utils:
    @staticmethod
    def scale_to_0_100(x, a, b):
        c = 0
        d = 100
        return int((x - a) * (d - c) / (b - a) + c)

    @staticmethod
    def scale_from_0_100(x, a, b):
        c = 0
        d = 100
        return int((x - c) * (b - a) / (d - c) + a)


class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Player")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)

        self.play_pause_button = QPushButton('Play/Pause')
        self.play_pause_button.clicked.connect(self.play_pause_video)

        self.choose_file_button = QPushButton('Choose File')
        self.choose_file_button.clicked.connect(self.chose_file_path)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.sliderMoved.connect(self.set_position)

        mode_layout = QHBoxLayout()

        self.video_mode_button = QRadioButton('Video')
        self.camera_mode_button = QRadioButton('Camera')
        self.video_mode_button.setChecked(True)
        self.video_mode_button.toggled.connect(self.set_video_mode)
        self.camera_mode_button.toggled.connect(self.set_camera_mode)

        mode_layout.addWidget(self.video_mode_button)
        mode_layout.addWidget(self.camera_mode_button)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.play_pause_button)
        hbox_layout.addWidget(self.slider)
        hbox_layout.addWidget(self.choose_file_button)

        vbox_layout = QVBoxLayout(self.central_widget)
        vbox_layout.addLayout(mode_layout)
        vbox_layout.addWidget(self.video_label)
        vbox_layout.addLayout(hbox_layout)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1000 // 30)  # 30 fps
        self.total_frames = 0
        self.is_playing = False
        self.mode = 0  # 0 - None, 1 - video path, 2 - camera
        self.file_path = None

    def set_video_mode(self):
        if self.mode == 2:
            self.stop_camera()
        pixmap = QPixmap(self.video_label.size())
        pixmap.fill(Qt.transparent)
        self.video_label.clear()
        self.choose_file_button.show()
        self.play_pause_button.show()
        self.slider.show()
        self.mode = 0
        self.slider.setValue(0)

    def set_camera_mode(self):
        pixmap = QPixmap(self.video_label.size())
        pixmap.fill(Qt.transparent)
        self.video_label.clear()
        self.play_pause_button.hide()
        self.choose_file_button.hide()
        self.slider.hide()
        self.timer.stop()
        self.mode = 2
        self.is_playing = True
        self.start_camera()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera
        self.timer.start(20)

    def stop_camera(self):

        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.video_label.clear()

    def update_frame(self):
        if self.is_playing:
            if self.mode == 1:
                ret, frame = self.cap.read()
                frame = self.search_face(frame)
                if ret:
                    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_img)
                    self.video_label.setPixmap(
                        pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                    current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                    self.slider.setValue(Utils.scale_to_0_100(current_frame, 0, self.total_frames))
            elif self.mode == 2:
                ret, frame = self.cap.read()
                if ret:
                    frame = self.search_face(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width, channel = frame.shape
                    bytes_per_line = 3 * width
                    q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(q_image))

    def search_face(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier('face.xml')
        result = faces.detectMultiScale(gray, 1.2, 2)
        for (x, y, w, h) in result:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

    def chose_file_path(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Media",
                                                  ".", "Video Files (*.mp4 *.flv *.ts *.mts *.avi)")
        if fileName != "":
            self.video_label.clear()
            self.slider.setValue(0)
            self.mode = 1
            self.file_path = fileName
            self.cap = cv2.VideoCapture(self.file_path)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText('Play')

    def play_pause_video(self):
        if self.is_playing and self.mode != 0:
            self.timer.stop()
            self.is_playing = False
            self.play_pause_button.setText('Play')
        elif self.mode != 0 and not self.is_playing:
            self.timer.start()
            self.is_playing = True
            self.play_pause_button.setText('Pause')

    def set_position(self, position):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, Utils.scale_from_0_100(position, 0, self.total_frames))

    def closeEvent(self, event):
        if self.cap:
            self.cap.release()

        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
