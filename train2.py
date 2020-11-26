import numpy as np
import pickle
import compress_pickle
from frame_data import FrameData
from frame_data import ContactData
from data_manager import DataManager
import os
import sys
import random
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
import math
import time
import multiprocessing

class History():
    def __init__(self):
        self.frames = []

    def updateFrame(self, frame):
        self.frames.append(frame)

    def getFeature(self, id):
        return feature
    
    def getKeyContact(self, frame): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add a member variable 'feature'
        DELAY = 5 # in frames
        contacts = []

        for contact in frame.contacts:
            if len(self.contacts[contact.id]) == DELAY or (len(self.contacts[contact.id]) < DELAY and contact.state == 3):
                feature = self.getFeature(contact.id)
                if len(feature) != 0:
                    contact.feature = feature
                    contacts.append(contact)

        return contacts

def input(user, session):
    X = []
    Y = []
    Z = []

    frames = pickle.load(open('data/' + user + '/' + str(session) + '.simple', 'rb'))

    history = History()
    for frame in frames:
        history.updateFrame(frame)
        key_contacts = history.getKeyContact(frame)
        for contact in key_contacts:
            if contact.label != -1:
                X.append(contact.feature)
                print(len(contact.feature))
                Y.append(contact.label)
                Z.append(user)
    
    return X, Y, Z

def para_input(user, session, queue):
    X, Y, Z = input(user, session)
    queue.put([X, Y, Z])

if __name__ == "__main__":
    start_time = time.perf_counter()

    file_name = DataManager(is_write=False).getFileName()
    tags = file_name.split('/')

    if tags[0] != 'xxx':
        users = [tags[0]]
    else:
        users = os.listdir('data/')

    if tags[1] != 'x':
        sessions = [int(tags[1])]
    else:
        sessions = [1, 2, 3, 4]
    
    X = []
    Y = []
    Z = [] # user_id
    queue = multiprocessing.Manager().Queue()
    jobs = []
    for user in users:
        for session in sessions:
            p = multiprocessing.Process(target=para_input, args=(user, session, queue, ))
            jobs.append(p)
            p.start()
    for p in jobs:
        p.join()
    for p in jobs:
        [xs, ys, zs] = queue.get()
        X.extend(xs)
        Y.extend(ys)
        Z.extend(zs)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    clf = svm.SVC(gamma='auto', class_weight='balanced')
    print('Positive samples = %d' % (np.sum(Y == 1)))
    print('Negative samples = %d' % (np.sum(Y == 0)))
    print('Accuracy = %f' % (np.mean(cross_val_score(clf, X, Y, cv=5))))

    print('Total time = %f' % (time.perf_counter() - start_time))

    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
