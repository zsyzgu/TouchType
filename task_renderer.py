import numpy as np
import random
import time
import cv2
import _thread
from abc import ABCMeta, abstractmethod

class TaskRenderer:
    def __init__(self):
        self.is_running = True
        self.is_entering = False
        self.label_id = 0
        try:
            _thread.start_new_thread(self._render, ())
        except:
            print("Thread Error")

    def __del__(self):
        self.is_running = False

    def _render(self):
        C, R = 500, 50

        while (self.is_running):
            image = np.zeros((R, C, 3))

            if self.is_entering:
                text = self._getCurrPhrase()
            else:
                text = 'Press Enter to Continue'
            image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if self.is_entering and self.label_id != 0:
                label_id = min(self.label_id, len(text))
                labeled_text = text[:label_id]
                image = cv2.putText(image, labeled_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if self.is_entering and self._isFinishPhrase():
                image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow('Task', image)
            cv2.waitKey(1)
    
    @abstractmethod
    def _isFinishPhrase(self):
        pass

    @abstractmethod
    def _getCurrPhrase(self):
        pass
