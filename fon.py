import cv2
import numpy as np
from tracker import *
from threading import Thread


def tracking(cap, nom):
    tracker = Tracker()
    alpha = 0.999  # Коэффициент для фоновой обработки
    FirstTime = True
    while True:
        ret, frame = cap.read()
    # Вычитание фона(кадр без людей) из кадра
        copy = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Устанавливаем первый кадр как фон, после изменяем фон с ипользованием смешивания
        if FirstTime:
            fon = copy
            FirstTime = False
        else:
            fon = cv2.addWeighted(copy, (1-alpha), fon, alpha, 0)
    # вычитаем фон и текущий кадр
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

        frame_copy = frame.copy()  # Копируем текущий кадр
        ids_boxes = tracker.update(detections, frame_copy)
    # Изображение рамки и (id)
        for ids in ids_boxes:
            x, y, w, h, id = ids
            cv2.putText(frame, str(id), (x, y - 15),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.imshow('Camera'+str(nom), frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


fon_4 = cv2.imread('fon4.png')
fon_4 = cv2.cvtColor(fon_4, cv2.COLOR_BGR2GRAY)
fon_3 = cv2.imread('fon.png')
fon_3 = cv2.cvtColor(fon_3, cv2.COLOR_BGR2GRAY)
camera1 = cv2.VideoCapture(
    "C:/Users/user/Desktop/st/3.Camera 2017-05-29 16-23-04_137 [3m3s].avi")
camera2 = cv2.VideoCapture(
    "C:/Users/user/Desktop/st/4.Camera 2017-05-29 16-23-04_137 [3m3s].avi")
Cam1 = Thread(target=tracking, args=(camera1, 3))
Cam2 = Thread(target=tracking, args=(camera2, 4))
Cam1.start()
Cam2.start()
cv2.destroyAllWindows()
