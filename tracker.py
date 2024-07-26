import math
import numpy as np
from scipy.optimize import linear_sum_assignment
import cv2


class Tracker:

    def __init__(self, distance_threshold=30):
        self.center_points = {}
        self.id_count = 0
        self.distance_threshold = distance_threshold
        self.id_deckriptor = []
        self.deskriptors = []

    def match(self, gray_cut):
        sift = cv2.SIFT_create()
        gray_cut = cv2.cvtColor(gray_cut, cv2.COLOR_BGR2GRAY)
        keypoints_1, descriptors_1 = sift.detectAndCompute(gray_cut, None)
        if len(self.id_deckriptor) == 0:
            self.deskriptors.append(descriptors_1)
            self.id_deckriptor.append(self.id_count)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            mach = []
            for desk in self.deskriptors:
                matches = flann.knnMatch(
                    descriptors_1, desk, k=2)
                mach.append(matches)
            max_value = max(mach)
            max_index = mach.index(max_value)
            self.id_deckriptor.append(self.id_deckriptor[max_index])
            self.deskriptors.append(descriptors_1)
            if self.id_deckriptor[len(self.id_deckriptor)-1] < self.id_count:
                self.id_count = self.id_deckriptor[len(self.id_deckriptor)-1]
            # return matches
        except:
            return 0

    def update(self, objects_rect, frame):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cut = frame[y:y+h, x:x+w]
            # cv2.imshow('3', cut)
            sopostavlenie = self.match(cut)
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                print(self.id_deckriptor[len(self.id_deckriptor)-1]-1 == id)
                if (dist < 100) and (self.id_deckriptor[len(self.id_deckriptor)-1]-1 == id):
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
                self.id_deckriptor[len(self.id_deckriptor)-1] = self.id_count
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids
