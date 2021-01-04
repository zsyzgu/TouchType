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

controller = Controller()

def pressAndRelease(key):
    controller.press(key)
    controller.release(key)

def typeEvent(contact):
    #print('[%f,%f],' % (contact.x, contact.y))
    data=[
    [0.135937,0.400060],
    [0.205401,0.383924],
    [0.275815,0.380078],
    [0.344803,0.370643],
    [0.414062,0.394802],
    [0.478125,0.389543],
    [0.563740,0.379297],
    [0.645211,0.373648],
    [0.708628,0.390144],
    [0.782863,0.367518],
    [0.158033,0.498377],
    [0.210309,0.473888],
    [0.290761,0.498407],
    [0.369107,0.519171],
    [0.443988,0.491797],
    [0.506165,0.495012],
    [0.587534,0.493089],
    [0.659528,0.507121],
    [0.721484,0.532602],
    [0.188145,0.625180],
    [0.262041,0.652644],
    [0.329518,0.609525],
    [0.400832,0.632843],
    [0.483560,0.644020],
    [0.545329,0.622957],
    [0.613077,0.635637],
    [0.095075,0.260156],
    [0.171094,0.251502],
    [0.236940,0.258263],
    [0.313077,0.259946],
    [0.385139,0.258714],
    [0.459018,0.261268],
    [0.532592,0.257392],
    [0.599287,0.263251],
    [0.670041,0.259195],
    [0.740132,0.253185],
    [0.807269,0.246905],
    [0.881760,0.257422],
    [0.941389,0.257963],
    [0.970924,0.396725],
    [0.843342,0.383444],
    [0.911719,0.375451],
    [0.788451,0.511118],
    [0.683849,0.631070],
    [0.757897,0.629718],
    [0.049440,0.633113],
    [0.882201,0.627764],
    [0.327055,0.751352],
    [0.602174,0.746124]]
    data = np.array(data)
    letter = 'qwertyuiopasdfghjklzxcvbnm1234567890-+BE[];,.SS__'
    min_dist2 = 1e9
    ret = ''
    for i in range(len(letter)):
        dist2 = (contact.x-data[i][0]) ** 2 + (contact.y-data[i][1]) ** 2
        if dist2 < min_dist2:
            min_dist2 = dist2
            ret = letter[i]
    
    if ret in 'qwertyuiopasdfghjklzxcvbnm1234567890-+[];,.':
        pressAndRelease(ret)
    if ret == 'B':
        pressAndRelease(Key.backspace)
    if ret == 'S':
        pressAndRelease(Key.shift)
    if ret == 'E':
        pressAndRelease(Key.enter)
    if ret == '_':
        pressAndRelease(Key.space)

if __name__ == "__main__":
    board = Board()
    keyboard = MyKeyboard()
    history = History()
    [scalar, clf] = pickle.load(open('model/tap.model', 'rb'))
    labels = np.zeros(20)

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
            if labels[contact.id] == 0 and pred == 1:
                sound.play()
                typeEvent(contact)
            labels[contact.id] = pred

        for contact in frame.contacts:
            if contact.state == 1:
                labels[contact.id] = 0
            contact.label = labels[contact.id]

        frame.output()
    
    cv2.destroyAllWindows()
    board.stop()
