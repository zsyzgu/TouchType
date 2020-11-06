import sys
import cv2
import time
import numpy as np
import pickle
import keyboard
from sklearn import svm
from board import Board
from task_manager import TaskManager

if __name__ == "__main__":
    board = Board()
    task_manager = TaskManager()

    is_running = True
    while (is_running):
        if keyboard.is_pressed('q'):
            is_running = False
        if keyboard.is_pressed('Enter'):
            if task_manager.isEntering(): # End task and save
                task = task_manager.getPhrase()
                succ = task_manager.endPhrase()
                if succ:
                    start_time = task_manager.start_time
                    end_time = task_manager.end_time
                    frames = board.get_frames_within(start_time, end_time)
                    word_begin_time = task_manager.word_begin_time
                    pickle.dump([task, word_begin_time, frames], open('data/' + str(task_manager.getPhraseId()) + '.pickle', 'wb'))
            else:
                succ = task_manager.startPhrase() # Start task

        frame = board.getFrame()
        frame.output()
        
        print(board.getFrameTime())
