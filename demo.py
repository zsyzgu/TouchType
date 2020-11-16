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
from train import History

if __name__ == "__main__":
    board = Board()
    keyboard = MyKeyboard()
    history = History()
    [scalar, clf] = pickle.load(open('model.pickle', 'rb'))
    labels = np.zeros(20)

    while True:
        if keyboard.is_pressed_down('Esc'):
            break
        frame = board.getFrame()
        history.updateFrame(frame)

        for contact in frame.contacts:
            if contact.state == 1:
                labels[contact.id] = 0
            if len(history.contacts[contact.id]) == 5 or contact.state == 3:
                feature = history.getFeature(contact.id)
                if len(feature) != 0:
                    feature = scalar.transform([feature])[0]
                    pred = clf.predict([feature])[0]
                    labels[contact.id] = pred
                    print(pred)
            contact.label = labels[contact.id]

        frame.output()
    
    cv2.destroyAllWindows()
    board.stop()
    