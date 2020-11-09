import sys
import cv2
import time
import numpy as np
import pickle
from sklearn import svm
from board import Board
import random
from task_renderer import TaskRenderer
from my_keyboard import MyKeyboard
from frame_data import FrameData

class ReplayTask(TaskRenderer):
    def __init__(self):
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self._mouseCallback)
        self.auto_play = False
        super().__init__()

    def __del__(self):
        super().__del__()
    
    def _isFinishPhrase(self):
        return self.is_entering and self._getCurrTimestamp() - self.start_time >= self.word_begin_time[-1] + 3

    def _getCurrPhrase(self):
        if self.is_entering:
            curr_phrase = ''
            t = self._getCurrTimestamp()
            words = self.task.split()
            for i in range(len(words)):
                if t - self.start_time >= self.word_begin_time[i]:
                    curr_phrase += words[i] + '_'
                else:
                    break
            return curr_phrase
        return ''
    
    def _getCurrTimestamp(self):
        return self.frames[self.frame_id].timestamp

    def _mouseCallback(self, event, x, y, flags, param):
        label = -1
        if event == cv2.EVENT_LBUTTONDOWN:
            label = 1
        if event == cv2.EVENT_RBUTTONDOWN:
            label = 0
        
        if label != -1:
            frame = self.frames[self.frame_id]
            (R, C) = frame.force_array.shape
            x = float(x) / (C * FrameData.MAGNI)
            y = float(y) / (R * FrameData.MAGNI)

            DIST_THRESHOLD = ((10 ** 2) / (R * C)) ** 0.5
            min_dist = DIST_THRESHOLD
            target = -1
            target_label = -1
            for contact in frame.contacts:
                dist = ((x - contact.x) ** 2 + (y - contact.y) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target = contact.id
                    target_label = contact.label
            
            if target != -1:
                if (label == 1) != (target_label == 1):
                    if label == 1:
                        self.incLabel()
                    else:
                        self.decLabel()
                
                for i in range(self.frame_id, -1, -1):
                    flag = False
                    for contact in self.frames[i].contacts:
                        if contact.id == target:
                            contact.label = label
                            flag = True
                    if not flag:
                        break

                for i in range(self.frame_id, len(self.frames)):
                    flag = False
                    for contact in self.frames[i].contacts:
                        if contact.id == target:
                            contact.label = label
                            flag = True
                    if not flag:
                        break

    def startTask(self):
        [task, word_begin_time, frames] = pickle.load(open('data/1.pickle', 'rb'))
        self.task = task
        self.word_begin_time = word_begin_time
        self.frame_id = 0
        self.frames = frames
        self.label_id = 0
        self.start_time = self.frames[0].timestamp
        self.is_entering = True

    def endTask(self):
        if self.is_entering == True and self._isFinishPhrase():
            self.is_entering = False
            pickle.dump([self.task, self.word_begin_time, self.frames], open('data/1_labeled.pickle', 'wb'))
    
    def playFrame(self):
        if self.is_entering:
            self.frames[self.frame_id].output()
    
    def incFrame(self):
        if self.is_entering and self.frame_id + 1 < len(self.frames):
            if len(self.frames[self.frame_id + 1].contacts) > len(self.frames[self.frame_id].contacts):
                self.auto_play = False
            self.frame_id += 1
    
    def decFrame(self):
        if self.is_entering and self.frame_id - 1 >= 0:
            self.frame_id -= 1
    
    def incLabel(self):
        if self.is_entering and self.label_id + 1 <= len(self._getCurrPhrase()):
            self.label_id += 1

    def decLabel(self):
        if self.is_entering and self.label_id - 1 >= 0:
            self.label_id -= 1

if __name__ == "__main__":
    replay_task = ReplayTask()
    keyboard = MyKeyboard()
    
    is_running = True
    while (is_running):
        frame_start_time = time.clock()

        if keyboard.is_pressed('q'):
            is_running = False
        if keyboard.is_pressed('left arrow') or keyboard.is_pressed('a'):
            replay_task.decFrame()
        if keyboard.is_pressed('right arrow') or keyboard.is_pressed('d'):
            replay_task.auto_play = True # Should we auto_play?
            replay_task.incFrame()
        elif replay_task.auto_play:
            replay_task.incFrame()
        if keyboard.is_pressed_down('Enter'):
            if replay_task.is_entering: # End task and save
                replay_task.endTask()
            else:
                replay_task.startTask()
        if keyboard.is_pressed_down('r'): # Redo:
            # TODO: set task_id
            replay_task.startTask()
        if keyboard.is_pressed_down('Space'):
            replay_task.incLabel()
        if keyboard.is_pressed_down('Backspace'):
            replay_task.decLabel()
        
        replay_task.playFrame()

        while ((time.clock() - frame_start_time) * Board.FPS < 1): # sync
            pass
