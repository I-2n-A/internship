import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


class Tracker:

    def __init__(self, distance_threshold=30):
        self.center_points = {}
        self.id_count = 0
        self.distance_threshold = distance_threshold
        self.deskriptors = []

    def match(self, gray_cut):
        sift = cv2.SIFT_create()
        gray_cut = cv2.cvtColor(gray_cut, cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = sift.detectAndCompute(gray_cut, None)
        self.deskriptors.append(descriptors_1)
        # print(len(self.deskriptors))
        MIN_MATCH_COUNT = 5
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = flann.knnMatch(
                descriptors_1, self.deskriptors[len(self.deskriptors)-1], k=2)
            print(len(matches))
            return matches
        except:
            return 0

    def update(self, objects_rect, frame):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cut = frame[y:y+h, x:x+w]
            cv2.imshow('3', cut)
            sopostavlenie = self.match(cut)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if (dist < 100) or (sopostavlenie != 0):
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
