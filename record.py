import sys
import cv2
import time
import numpy as np
import pickle
import keyboard
from sklearn import svm
from board import Board

if __name__ == "__main__":
    board = Board()
    is_running = True

    while (is_running):
        if keyboard.is_pressed('q'):
            is_running = False

        frame = board.getFrame()
        frame.output()
        frames = board.frames

        if len(frames) >= 2:
            print(frames[-1].timestamp - frames[-2].timestamp)
    board.save('data/test/pickle')
