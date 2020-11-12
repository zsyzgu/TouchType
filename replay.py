import sys
import cv2
import time
import numpy as np
import pickle
import compress_pickle
from sklearn import svm
from board import Board
import random
import threading
from my_keyboard import MyKeyboard
from frame_data import FrameData
from data_manager import DataManager

class Replay():
    SCREENSHOT_W = 980
    SCREENSHOT_H = 540

    def __init__(self, file_name):
        self.file_name = file_name
        self.init()
        thread = threading.Thread(target=self._run)
        thread.start()

    def _run(self):
        cv2.namedWindow('screenshot')
        cv2.setMouseCallback('screenshot', self._screenshotMouseCallback)
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self._frameMouseCallback)

        self.is_running = True
        while self.is_running:
            self._renderScreenshot()
            self.frames[self.frame_id].output()
            if self.auto_play:
                self.incFrame()
    
    def _renderScreenshot(self):
        W = Replay.SCREENSHOT_W
        H = Replay.SCREENSHOT_H
        screenshot = self._getScreenshot(self.frames[self.frame_id].timestamp)
        screenshot = cv2.resize(screenshot, (W, H))
        cv2.line(screenshot, (0, int(0.9*H)), (W-1, int(0.9*H)), (192, 192, 192), 3)
        schedule = float(self.frame_id) / len(self.frames)
        cv2.rectangle(screenshot, (int(schedule*W-2), int(0.9*H-10)), (int(schedule*W+2), int(0.9*H+10)), (255, 255, 255), -1)
        cv2.imshow('screenshot', screenshot)
    
    def _getScreenshot(self, timestamp):
        for index in range(len(self.timestamps)):
            if timestamp < self.timestamps[index]:
                break
        while (index >= len(self.screenshots)):
            succ, image = self.capture.read()
            self.screenshots.append(image)
        image = self.screenshots[index]
        return image

    def _screenshotMouseCallback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN or flags == cv2.EVENT_FLAG_LBUTTON:
            x = float(x) / (Replay.SCREENSHOT_W)
            y = float(y) / (Replay.SCREENSHOT_H)
            if (0.8 <= y and y <= 1.0) and (0.0 <= x and x <= 1.0):
                self.frame_id = int(x * (len(self.frames) - 1))

    def _frameMouseCallback(self, event, x, y, flags, param):
        label = None
        if event == cv2.EVENT_LBUTTONDOWN:
            label = 1
        if event == cv2.EVENT_RBUTTONDOWN:
            label = 0
        if event == cv2.EVENT_MBUTTONDOWN:
            label = -1
        
        if label !=None:
            frame = self.frames[self.frame_id]
            (R, C) = frame.force_array.shape
            x = float(x) / (C * FrameData.MAGNI)
            y = float(y) / (R * FrameData.MAGNI)

            DIST_THRESHOLD = ((10 ** 2) / (R * C)) ** 0.5
            min_dist = DIST_THRESHOLD
            target = -1
            for contact in frame.contacts:
                dist = ((x - contact.x) ** 2 + (y - contact.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target = contact.id
            
            if target != -1:
                for i in range(self.frame_id, -1, -1):
                    flag = False
                    for contact in self.frames[i].contacts:
                        if contact.id == target:
                            contact.label = label
                            flag = (contact.state != 1)
                    if not flag:
                        break

                for i in range(self.frame_id, len(self.frames)):
                    flag = False
                    for contact in self.frames[i].contacts:
                        if contact.id == target:
                            contact.label = label
                            flag = (contact.state != 3)
                    if not flag:
                        break

    def init(self):
        self.capture = cv2.VideoCapture('data/' + self.file_name + '.avi')
        self.screenshots = []
        self.timestamps = pickle.load(open('data/' + self.file_name + '.timestamp', 'rb'))
        self.frames = compress_pickle.load('data/' + self.file_name + '.gz')
        self.frame_id = 0
        self.auto_play = False

    def stop(self):
        self.is_running = False
        file_path = 'data/' + self.file_name + '_labeled.gz'
        compress_pickle.dump(self.frames, file_path)
        # TODO: confirm if there exist a file

    def incFrame(self):
        if self.frame_id + 1 < len(self.frames):
            self.frame_id += 1

    def decFrame(self):
        if self.frame_id - 1 >= 0:
            self.frame_id -= 1
    
    def setAutoPlay(self):
        self.auto_play ^= True

if __name__ == "__main__":
    keyboard = MyKeyboard()
    file_name = DataManager(is_write=False).getFileName()
    replay = Replay(file_name)
    
    while True:
        frame_start_time = time.perf_counter()

        if keyboard.is_pressed('q'):
            replay.is_running = False
            break    
        if keyboard.is_pressed('left arrow') or keyboard.is_pressed('a'):
            replay.decFrame()
        if keyboard.is_pressed('right arrow') or keyboard.is_pressed('d'):
            replay.incFrame()
        if keyboard.is_pressed_down('r'): # Redo:
            replay.init()
        if keyboard.is_pressed_down('Space'): # Set auto play
            replay.setAutoPlay()
        
        while ((time.perf_counter() - frame_start_time) * Board.FPS < 1): # sync
            time.sleep(0.005)
    
    replay.stop()
