import sys
import cv2
import time
import numpy as np
import pickle
import keyboard
from sklearn import svm
from board import Board
from history import History

if __name__ == "__main__":
    board = Board()
    history = History()
    [scalar, clf] = pickle.load(open('model.pickle', 'rb'))
    is_running = True

    is_contact = [0 for i in range(20)]
    while (is_running):
        frame = board.getFrame()

        history.update_frame(frame)
        for contact in frame.contacts:
            if contact.state == 1:
                is_contact[contact.id] = 0
            if len(history.contacts[contact.id]) == 3:
                feature = history.get_feature(contact.id)
                feature = scalar.transform([feature])[0]
                if clf.predict([feature])[0] == 1:
                    is_contact[contact.id] = 1
        
        real_contacts = []
        for contact in frame.contacts:
            if is_contact[contact.id]:
                real_contacts.append(contact)
        
        frame.output(real_contacts)
        frames = board.frames

        if len(frames) >= 2:
            print(frames[-1].timestamp - frames[-2].timestamp)
        if keyboard.is_pressed('q'):
            is_running = False
    pickle.dump(frames, open('data/test.pickle', 'wb'))
        
    '''
    frames = pickle.load(open('data/P_1.pickle', 'rb'))
    for frame_data in frames:
        time.sleep(0.008)
        frame_data.output()
    '''

