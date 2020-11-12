import sys
import cv2
import time
import numpy as np

class ContactData():
    def __init__(self, id = 0, state = 0, x = 0, y = 0, area = 0, force = 0, major = 0, minor = 0, delta_x = 0, delta_y = 0, delta_force = 0, delta_area = 0, label = 0):
        self.id = id
        self.state = state
        self.x = x
        self.y = y
        self.area = area
        self.force = force
        self.major = major
        self.minor = minor
        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta_force = delta_force
        self.delta_area = delta_area
        self.label = label

class FrameData():
    MAGNI = 3

    def __init__(self, force_array, timestamp):
        self.force_array = force_array
        self.timestamp = timestamp
        self.contacts = []
    
    def append_contact(self, contact):
        self.contacts.append(contact)

    def output(self):
        MAGNI = FrameData.MAGNI
        (R, C) = self.force_array.shape
        force_map = cv2.resize(self.force_array, (C * MAGNI, R * MAGNI)).astype(np.float32)
        force_map = cv2.cvtColor(force_map, cv2.COLOR_GRAY2BGR)
        for contact in self.contacts:
            color = (0, 0, 0)
            if contact.label == -1: # Unsure
                color = (255, 0, 0)
            if contact.label == 1: # Positive sample
                color = (0, 255, 0)
            if contact.label == 0: # Negative sample
                color = (0, 0, 255)
            cv2.circle(force_map, (int((contact.x * C + 0.5) * MAGNI), int((contact.y * R + 0.5) * MAGNI)), 4, color, -1)
        cv2.imshow('frame', force_map)
        cv2.waitKey(1)