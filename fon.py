import cv2
import numpy as np
from PIL import Image

cap = cv2.VideoCapture(
    "C:/Users/user/Desktop/st/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi")
fon = cv2.imread('fon.png')
fon = cv2.cvtColor(fon, cv2.COLOR_BGR2GRAY)

while True:
    ret, frame = cap.read()
# Вычитание фона(кадр без людей) из кадра
    copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dframe = cv2.absdiff(copy, fon)
    th, dframe = cv2.threshold(dframe, 35, 255, cv2.THRESH_BINARY)
# Обнаружение контуров человека
    _, thresh = cv2.threshold(dframe, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    if len(contours) != 0:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 100:
            x, y, w, h = cv2.boundingRect(largest_contour)
            detections.append([x, y, w, h])
# Изображение рамки
    for cnt in detections:
        x, y, w, h = cnt
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    cv2.imshow('frame', dframe)
    cv2.imshow('Camera-1', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
