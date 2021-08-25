import cv2
import numpy as np
from datetime import datetime
import json
import imutils
import queue

class MotionDetector:

    font = cv2.FONT_HERSHEY_PLAIN
    comparison_frame = None
    minimum_area: int = None

    def __init__(self, first_frame, queue_size, mot_thresh,  minimum_area=400, ):
        self.minimum_area = minimum_area
        self.mot_thresh = mot_thresh
        self.frame_queue = queue.Queue()
        grayscale = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        for i in range(queue_size):
            self.frame_queue.put(grayscale)

    def blur_frame(self, frame):
        return cv2.GaussianBlur(frame, (21, 21), 0)

    def detect(self, frame):
        # compute the absolute difference between the current frame and
        # comparison frame

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        comparison_frame = self.frame_queue.get()
        frame_delta = cv2.absdiff(self.blur_frame(comparison_frame), self.blur_frame(grayscale))
        thresh = cv2.threshold(frame_delta, self.mot_thresh, 255, cv2.THRESH_BINARY)[1]

        # dilate the thresholded image to fill in the holes, then find
        # contours on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours,_ = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours = imutils.grab_contours(contours)

        # loop over the contours
        bboxes = []
        for index, contour in enumerate(contours):
            # if the contour is too small, ignore it
            x,y,w,h = cv2.boundingRect(contour)
            if w*h > self.minimum_area:
                bboxes.append((x,y,x+w,y+h))

        self.frame_queue.put(grayscale)
        return bboxes

