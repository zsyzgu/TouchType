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
import pygame
from pynput.keyboard import Key, Controller
from layout import Layout

if __name__ == "__main__":
    board = Board()
    keyboard = MyKeyboard()
    history = History()
    [scalar, clf] = pickle.load(open('model/tap.model', 'rb'))
    labels = [None for i in range(20)]
    controller = Controller()
    qwerty = Layout()

    pygame.mixer.init(22050, -16, 2, 64)
    pygame.init()
    sound = pygame.mixer.Sound("sound/type.wav")
    sound.set_volume(1.0)

    while True:
        if keyboard.is_pressed_down('Esc'):
            break

        frame = board.getNewFrame()
        history.updateFrame(frame)
        
        key_contacts = history.getKeyContact(frame)
        for contact in key_contacts:
            feature = scalar.transform([contact.feature])[0]
            pred = clf.predict([feature])[0]
            if labels[contact.id] == None and pred == 1:
                sound.play()
                key = qwerty.decode(contact.x, contact.y)
                controller.press(key)
                labels[contact.id] = key

        for contact in frame.contacts:
            if contact.state == 1:
                labels[contact.id] = None
            if contact.state == 3:
                if labels[contact.id] != None:
                    controller.release(labels[contact.id])
            contact.label = (labels[contact.id] != None)

        #frame.output()
    
    #cv2.destroyAllWindows()
    board.stop()


