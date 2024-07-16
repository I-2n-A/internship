import cv2
import numpy as np
import math
from PIL import Image
import os
import matplotlib.pyplot as plt


def sif(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # sift
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(img1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1, descriptors_2)
    matches = sorted(matches, key=lambda x: x.distance)

    img3 = cv2.drawMatches(img1, keypoints_1, img2,
                           keypoints_2, matches[:50], img2, flags=2)
    plt.imshow(img3), plt.show()


cap = cv2.VideoCapture(
    "C:/Users/user/Desktop/st/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi")
backSub = cv2.createBackgroundSubtractorMOG2()
r, h, c, w = 200, 20, 300, 20
track_window = (c, r, w, h)
while True:
    ret, frame = cap.read()
    frame_copy = frame.copy()
    frame_copy[60:110, 600:660] = 127
    rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    blured = cv2.GaussianBlur(rgb, (11, 11), 0, 0)
    gray = cv2.cvtColor(blured, cv2.COLOR_RGB2GRAY)
    fg_mask = backSub.apply(gray)
    _, thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    if len(contours) != 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            detections.append([x, y, w, h])
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)),
                       np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    ret, track_window = cv2.meanShift(thresh, track_window, term_crit)
    x, y, w, h = track_window
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(gray, None)
    for cnt in detections:
        x, y, w, h = cnt
        crop = frame[y:y+h, x:x+w]
        # sif(frame, crop)
        frame = cv2.drawKeypoints(
            frame, keypoints_1, frame,  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame_out = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow('Camera-1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
