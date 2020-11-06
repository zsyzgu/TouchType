import numpy as np
import random
import time
import cv2
import _thread

class TaskManager:
    WORD_TIME_MEAN = 2.5
    WORD_TIME_STD = 1.5
    START_PHRASE_TIME = 3.0
    END_PHRASE_TIME = 5.0
    TOTAL_TASK = 5

    def __init__(self):
        lines = open('phrases.txt', 'r').readlines()
        self.phrases = [line.strip().lower() for line in lines]
        random.shuffle(self.phrases)
        self.is_running = True
        self.is_entering = False
        self.phrase_id = 0
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
            color = (255, 255, 255)
            if self.is_entering:
                text = self.getCurrPhrase()
                if self._is_finish_phrase():
                    color = (0, 0, 255)
            else:
                text = 'Press Enter to Continue'
            image = cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.imshow('Task', image)
            cv2.waitKey(1)
    
    def _is_finish_phrase(self):
        return self.is_entering and time.clock() - self.start_time >= self.word_begin_time[-1] + TaskManager.END_PHRASE_TIME

    def startPhrase(self):
        if self.is_entering == False:
            self.is_entering = True
            self.start_time = time.clock()
            phrase = self.getPhrase()
            words = phrase.split()
            word_duration = np.random.normal(TaskManager.WORD_TIME_MEAN, TaskManager.WORD_TIME_STD, len(words))
            self.word_begin_time = np.cumsum(word_duration) - word_duration[0] + TaskManager.START_PHRASE_TIME
            return True
        return False

    def endPhrase(self):
        if self.is_entering == True and self._is_finish_phrase():
            self.is_entering = False
            self.phrase_id += 1
            self.end_time = time.clock()
            if self.phrase_id >= TaskManager.TOTAL_TASK:
                self.is_running = False
            return True
        return False

    def getPhrase(self):
        return self.phrases[self.phrase_id]

    def getCurrPhrase(self):
        curr_phrase = ''
        t = time.clock()
        phrase = self.getPhrase()
        words = phrase.split()
        for i in range(len(words)):
            if t - self.start_time >= self.word_begin_time[i]:
                curr_phrase += words[i] + ' '
            else:
                break
        return curr_phrase

    def getPhraseId(self):
        return self.phrase_id
    
    def isEntering(self):
        return self.is_entering
