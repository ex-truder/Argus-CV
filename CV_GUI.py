import imutils
from imutils import perspective, contours
from PyQt5 import QtCore, QtGui, QtWidgets
import cv2
from matplotlib import pyplot as plt
from mss import mss
from PIL import Image
import sys
import numpy as np
from scipy.spatial import distance as dist

class ArgusCV(object):
    def setupUi(self, MainWindow):

        self.frameset = []

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1150, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setWindowTitle("CV")

        # FONT SETUP
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)

        # ERODE
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName(u"lineEdit")
        self.lineEdit.setGeometry(QtCore.QRect(130, 270, 113, 20))

        # SLIDER LOW
        self.sl_thres_low = QtWidgets.QSlider(self.centralwidget)
        self.sl_thres_low.setGeometry(QtCore.QRect(120, 720, 681, 22))
        self.sl_thres_low.setMinimum(0)
        self.sl_thres_low.setMaximum(255)
        self.sl_thres_low.setSingleStep(0)
        self.sl_thres_low.setValue(140)
        self.sl_thres_low.setOrientation(QtCore.Qt.Horizontal)
        self.sl_thres_low.setObjectName("sl_thres_low")

        self.label_thres_low = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_low.setGeometry(QtCore.QRect(90, 720, 21, 21))
        self.label_thres_low.setObjectName("label_thres_low")

        self.label_thres_low_val = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_low_val.setGeometry(QtCore.QRect(820, 720, 21, 21))
        self.label_thres_low_val.setObjectName("label_thres_low_val")
        self.label_thres_low.setText("Low")
        self.label_thres_low_val.setText("0")

        # SLIDER HI
        self.sl_thres_hi = QtWidgets.QSlider(self.centralwidget)
        self.sl_thres_hi.setGeometry(QtCore.QRect(120, 740, 681, 22))
        self.sl_thres_hi.setMinimum(1)
        self.sl_thres_hi.setMaximum(255)
        self.sl_thres_hi.setSingleStep(0)
        self.sl_thres_hi.setValue(255)
        self.sl_thres_hi.setOrientation(QtCore.Qt.Horizontal)
        self.sl_thres_hi.setObjectName("sl_thres_hi")

        self.label_thres_hi = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_hi.setGeometry(QtCore.QRect(90, 740, 21, 21))
        self.label_thres_hi.setObjectName("label_blur")

        self.label_thres_hi_val = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_hi_val.setGeometry(QtCore.QRect(820, 740, 21, 21))
        self.label_thres_hi_val.setObjectName("label_thres_hi_val")
        self.label_thres_hi.setText("High")
        self.label_thres_hi_val.setText("0")

        # SLIDER BLUR
        self.sl_thres_blur = QtWidgets.QSlider(self.centralwidget)
        self.sl_thres_blur.setGeometry(QtCore.QRect(120, 700, 681, 22))
        self.sl_thres_blur.setMinimum(1)
        self.sl_thres_blur.setMaximum(200)
        self.sl_thres_blur.setSingleStep(0)
        self.sl_thres_blur.setValue(3)
        self.sl_thres_blur.setOrientation(QtCore.Qt.Horizontal)
        self.sl_thres_blur.setObjectName("sl_blur")

        self.label_thres_blur = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_blur.setGeometry(QtCore.QRect(90, 700, 21, 21))
        self.label_thres_blur.setObjectName("label_blur")

        self.label_thres_blur_val = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_blur_val.setGeometry(QtCore.QRect(820, 700, 21, 21))
        self.label_thres_blur_val.setObjectName("label_thres_hi_val")
        self.label_thres_blur.setText("Blur")
        self.label_thres_blur_val.setText("0")

        # COMMENT UI
        self.label_thres_hi_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_thres_hi_2.setGeometry(QtCore.QRect(20, 720, 61, 31))
        self.label_thres_hi_2.setFont(font)
        self.label_thres_hi_2.setObjectName("label_thres_hi_2")
        self.label_thres_hi_2.setText("Threshold")

        # MAIN PICTURE
        self.pic01 = QtWidgets.QLabel(self.centralwidget)
        self.pic01.setGeometry(QtCore.QRect(10, 10, 1100, 640))
        self.pic01.setFrameShape(QtWidgets.QFrame.Box)
        self.pic01.setLineWidth(1)
        self.pic01.setText("")
        self.pic01.setScaledContents(False)
        self.pic01.setAlignment(QtCore.Qt.AlignCenter)
        self.pic01.setWordWrap(False)
        self.pic01.setObjectName("pic01")

        # MISCELLANEOUS PICTURE 02
        self.pic02 = QtWidgets.QLabel(self.centralwidget)
        self.pic02.setGeometry(QtCore.QRect(1120, 10, 1100 // 2, 640 // 2))
        self.pic02.setFrameShape(QtWidgets.QFrame.Box)
        self.pic02.setLineWidth(1)
        self.pic02.setText("")
        self.pic02.setScaledContents(False)
        self.pic02.setAlignment(QtCore.Qt.AlignCenter)
        self.pic02.setWordWrap(False)
        self.pic02.setObjectName("pic02")

        # MISCELLANEOUS PICTURE 03
        self.pic03 = QtWidgets.QLabel(self.centralwidget)
        self.pic03.setGeometry(QtCore.QRect(1120, 660 // 2, 1100 // 2, 640 // 2))
        self.pic03.setFrameShape(QtWidgets.QFrame.Box)
        self.pic03.setLineWidth(1)
        self.pic03.setText("")
        self.pic03.setScaledContents(False)
        self.pic03.setAlignment(QtCore.Qt.AlignCenter)
        self.pic03.setWordWrap(False)
        self.pic03.setObjectName("pic03")

        MainWindow.setCentralWidget(self.centralwidget)

        self.sl_thres_low.sliderMoved['int'].connect(self.label_thres_low_val.setNum)
        self.sl_thres_hi.sliderMoved['int'].connect(self.label_thres_hi_val.setNum)
        self.sl_thres_blur.sliderMoved['int'].connect(self.label_thres_blur_val.setNum)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # TODO: Gather this shit up
    def cv2_works(self, cam, l_thres, h_thres, blur):
        img = self.estimate_average(cam, 10)
        new_y = 600
        if blur % 2:
            img_in = cv2.GaussianBlur(img, (blur, blur), 0)
        else:
            img_in = cv2.GaussianBlur(img, (blur + 1, blur + 1), 0)
        img_in = cv2.resize(img_in, (int(new_y * (max(img_in.shape[0:2]) / min(img_in.shape[0:2]))), new_y),
                            interpolation=cv2.INTER_CUBIC)
        img = cv2.cvtColor(img_in, cv2.COLOR_RGB2GRAY)

        # THRESHOLDING IMAGE
        # TODO Investigate difference between cv2.threshold & cv2.Canny
        # T, thresh_img = cv2.threshold(img, l_thres, 255, cv2.THRESH_BINARY)
        thresh_img = cv2.Canny(img, l_thres, h_thres)
        thresh_img = cv2.dilate(thresh_img, None, iterations=1)
        thresh_img = cv2.erode(thresh_img, None, iterations=1)

        # FOR COLORED IMAGING
        # h, w, ch = thresh_img.shape
        # thresh_img = QtGui.QImage(thresh_img, w, h, w * ch, QtGui.QImage.Format_RGB888).rgbSwapped()

        # INVERTING IMAGE
        # thresh_img = cv2.bitwise_not(thresh_img)

        # FINDING CONTOURS
        contours = self.find_contours(thresh_img)

        # FINDING KeyPoints
        kp = self.find_features(thresh_img)

        actual_contours = 0

        # compute the rotated bounding box of the contour
        for c in contours:
            if cv2.contourArea(c) < 600:
                continue
            actual_contours += 1
            box = cv2.minAreaRect(c)
            box = cv2.boxPoints(box)
            box = np.array(box, dtype="int")
            box = perspective.order_points(box)
            cv2.drawContours(img_in, [box.astype("int")], -1, (0, 255, 0), 1)
            # loop over the original points and draw them
            for (x, y) in box:
                cv2.circle(img_in, (int(x), int(y)), 5, (0, 0, 255), -1)

            # unpack the ordered bounding box, then compute the midpoint
            # between the top-left and top-right coordinates, followed by
            # the midpoint between bottom-left and bottom-right coordinates
            (tl, tr, br, bl) = box
            (tltrX, tltrY) = self.midpoint(tl, tr)
            (blbrX, blbrY) = self.midpoint(bl, br)

            # compute the midpoint between the top-left and top-right points,
            # followed by the midpoint between the top-righ and bottom-right
            (tlblX, tlblY) = self.midpoint(tl, bl)
            (trbrX, trbrY) = self.midpoint(tr, br)

            # draw the midpoints on the image
            cv2.circle(img_in, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
            cv2.circle(img_in, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
            cv2.circle(img_in, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
            cv2.circle(img_in, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

            # draw lines between the midpoints
            cv2.line(img_in, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
                     (255, 0, 255), 1)
            cv2.line(img_in, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
                     (255, 0, 255), 1)

            # compute the Euclidean distance between the midpoints
            dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
            dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

            # draw the object sizes on the image
            cv2.putText(img_in, "{:.1f}px".format(dA),
                        (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (50, 200, 0), 2)
            cv2.putText(img_in, "{:.1f}px".format(dB),
                        (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (50, 200, 0), 2)
        cv2.putText(img_in, "Total:{:d}".format(actual_contours),
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (240, 20, 0), 2)

            # cv2.drawContours(img_in, contours, -1, (0, 0, 255), 1)
        # img_in = cv2.drawKeypoints(img_in, kp, None, color=(0, 255, 0), flags=0)
        h, w, d = img_in.shape
        img_in = QtGui.QImage(img_in, w, h, d * w, QtGui.QImage.Format_RGB888)
        self.pic01.setPixmap(QtGui.QPixmap.fromImage(img_in))
        thresh_img = imutils.resize(thresh_img, width=w//2)
        h1, w1 = thresh_img.shape
        thresh_img = QtGui.QImage(thresh_img, w1, h1, w1, QtGui.QImage.Format_Grayscale8)
        self.pic02.setPixmap(QtGui.QPixmap.fromImage(thresh_img))

        if cv2.waitKey(2) & 0xFF == ord('q'):
            print("Q")

    def find_contours(self, img):
        # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = imutils.grab_contours(contours)
        return contours

    def find_features(self, img):
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img, None)
        return kp

    def camera_init(self, desc):
        cap = cv2.VideoCapture(desc, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 30.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, -5.0)
        cap.set(cv2.CAP_PROP_CONTRAST, 3.0)
        return cap

    def get_camera_frame(self, cam):
        frame = []
        if cam.isOpened():
            ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = imutils.resize(frame, width=600)
        return frame

    def estimate_average(self, cam, depth):
        frame = self.get_camera_frame(cam)
        if len(self.frameset) == depth:
            self.frameset.pop(0)
        while len(self.frameset) < depth:
            self.frameset.append(frame)
        for i in range(len(self.frameset)):
            if i == 0:
                pass
            else:
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                frame = cv2.addWeighted(self.frameset[i], alpha, frame, beta, 0.0)
        # result = self.get_camera_frame(cam)
        return frame

    def midpoint(self, ptA, ptB):
        return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5

    def exec(self):
        sys.exit(app.exec_())


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    inspector = ArgusCV()
    inspector.setupUi(MainWindow)
    MainWindow.show()
    cam = inspector.camera_init(0)
    while True:
        inspector.cv2_works(cam, inspector.sl_thres_low.value(),
                            inspector.sl_thres_hi.value(), inspector.sl_thres_blur.value())
        print("Working")
        if cv2.waitKey(20) & 0xFF == ord('q'):
            print("Q")
            break
    sys.exit(app.exec_())
