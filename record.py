import sys
import cv2
import time
import numpy as np
import pickle
import compress_pickle
from PIL import ImageGrab
from board import Board
import random
from my_keyboard import MyKeyboard
import multiprocessing
from data_manager import DataManager

def record_screen(is_end, file_name):
    FPS = 20

    screenshot = ImageGrab.grab()
    H, W = screenshot.size
    video = cv2.VideoWriter('data/' + file_name + '.avi', cv2.VideoWriter_fourcc(*'XVID'), FPS, (H, W))
    timestamps = []

    while is_end.qsize() == 0:
        screenshot = ImageGrab.grab()
        frame = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        video.write(frame)
        timestamps.append(time.perf_counter())
    
    pickle.dump(timestamps, open('data/' + file_name + '.timestamp', 'wb'))
    video.release()
    print('Video released.', time.perf_counter())

def record_board(is_end, file_name):
    board = Board()

    while is_end.qsize() == 0:
        frame = board.getFrame()
        frame.output()
        #print(board.getFrameTime())
    cv2.destroyAllWindows()
    
    board.stop()
    print('Board compressing.', time.perf_counter())
    compress_pickle.dump(board.frames, 'data/' + file_name + '.gz')
    print('Board released.', time.perf_counter())

if __name__ == "__main__":
    file_name = DataManager().getFileName()
    is_end_1 = multiprocessing.Queue()
    is_end_2 = multiprocessing.Queue()
    p1 = multiprocessing.Process(target=record_screen, args=(is_end_1, file_name, ))
    p2 = multiprocessing.Process(target=record_board, args=(is_end_2, file_name, ))
    p1.start()
    p2.start()
    
    keyboard = MyKeyboard()
    while True:
        if keyboard.is_pressed('Esc'):
            is_end_1.put(1)
            is_end_2.put(1)
            break
        time.sleep(0.05)
    
    p1.join()
    p2.join()
