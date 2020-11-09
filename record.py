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

class RecordTask(TaskRenderer):
    WORD_TIME_MEAN = 2.5
    WORD_TIME_STD = 1.5
    START_PHRASE_TIME = 3.0
    END_PHRASE_TIME = 5.0
    TOTAL_TASK = 5

    def __init__(self):
        lines = open('phrases.txt', 'r').readlines()
        self.phrases = [line.strip().lower() for line in lines]
        random.shuffle(self.phrases)
        self.phrase_id = 0
        super().__init__()

    def __del__(self):
        super().__del__()
    
    def _isFinishPhrase(self):
        return self.is_entering and time.clock() - self.start_time >= self.word_begin_time[-1] + RecordTask.END_PHRASE_TIME

    def _getCurrPhrase(self):
        if self.is_entering:
            curr_phrase = ''
            t = time.clock()
            phrase = self.getPhrase()
            words = phrase.split()
            for i in range(len(words)):
                if t - self.start_time >= self.word_begin_time[i]:
                    curr_phrase += words[i] + '_'
                else:
                    break
            return curr_phrase
        return ''

    def startPhrase(self):
        self.is_entering = True
        self.start_time = time.clock()
        phrase = self.getPhrase()
        words = phrase.split()
        word_duration = np.random.normal(RecordTask.WORD_TIME_MEAN, RecordTask.WORD_TIME_STD, len(words))
        self.word_begin_time = np.cumsum(word_duration) - word_duration[0] + RecordTask.START_PHRASE_TIME

    def endPhrase(self):
        if self.is_entering == True and self._isFinishPhrase():
            self.is_entering = False
            self.phrase_id += 1
            self.end_time = time.clock()
            if self.phrase_id >= RecordTask.TOTAL_TASK:
                self.is_running = False
            return True
        return False

    def getPhrase(self):
        return self.phrases[self.phrase_id]

    def getPhraseId(self):
        return self.phrase_id

if __name__ == "__main__":
    board = Board()
    record_task = RecordTask()
    keyboard = MyKeyboard()
    
    is_running = True
    while (is_running):
        if keyboard.is_pressed('q'):
            is_running = False
        if keyboard.is_pressed_down('Enter'):
            if record_task.is_entering: # End task and save
                task = record_task.getPhrase()
                succ = record_task.endPhrase()
                if succ:
                    start_time = record_task.start_time
                    end_time = record_task.end_time
                    frames = board.get_frames_within(start_time, end_time)
                    word_begin_time = record_task.word_begin_time
                    pickle.dump([task, word_begin_time, frames], open('data/' + str(record_task.getPhraseId()) + '.pickle', 'wb'))
            else:
                record_task.startPhrase() # Start task
                print('Phrase ID =', record_task.phrase_id)
        if keyboard.is_pressed_down('r'): # Redo
            if not record_task.is_entering and record_task.phrase_id - 1 >= 0:
                record_task.phrase_id -= 1
            record_task.startPhrase() # Start task
            print('Phrase ID =', record_task.phrase_id)

        frame = board.getFrame()
        frame.output()
        
        print(board.getFrameTime())
