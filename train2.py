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
from board import Board
import cv2

class History():
    def __init__(self):
        self.N = 16
        self.frames = []
        self.start_frame = np.zeros(self.N)

    def updateFrame(self, frame):
        self.frames.append(frame)

        for contact in frame.contacts:
            if contact.state == 1:
                self.start_frame[contact.id] = len(self.frames)

    def getFeature(self, id):
        T = 1.0
        D = 1.0
        N = 20
        M = 10
        feature = []

        for contact in self.frames[-1].contacts:
            if contact.id == id:
                key_contact = contact

        force_map = np.zeros((N + 1, M + 1))
        area_map = np.zeros((N + 1, M + 1))
        inten_map = np.zeros((N + 1, M + 1))
        ell_map = np.zeros((N + 1, M + 1))

        for fid in range(1, int(T * Board.FPS) + 1):
            if fid < len(self.frames):
                for contact in self.frames[-fid].contacts:
                    t = (float(fid) / (Board.FPS * T))
                    d = (((key_contact.x - contact.x) ** 2 + (key_contact.y - contact.y) ** 2) * 0.5) ** 0.5
                    d = d / D
                    t = min(t * N, N - 0.001)
                    d = min(d * M, M - 0.001)
                    ti = int(t)
                    di = int(d)
                    td = t - ti
                    dd = d - di
                    force_map[ti, di] += contact.force * (1 - td) * (1 - dd)
                    force_map[ti + 1, di] += contact.force * td * (1 - dd)
                    force_map[ti, di + 1] += contact.force * (1 - td) * dd
                    force_map[ti + 1, di + 1] += contact.force * td * dd
                    area_map[ti, di] += contact.area * (1 - td) * (1 - dd)
                    area_map[ti + 1, di] += contact.area * td * (1 - dd)
                    area_map[ti, di + 1] += contact.area * (1 - td) * dd
                    area_map[ti + 1, di + 1] += contact.area * td * dd
                    inten_map[ti, di] += (contact.force / contact.area) * (1 - td) * (1 - dd)
                    inten_map[ti + 1, di] += (contact.force / contact.area) * td * (1 - dd)
                    inten_map[ti, di + 1] += (contact.force / contact.area) * (1 - td) * dd
                    inten_map[ti + 1, di + 1] += (contact.force / contact.area) * td * dd
                    if contact.major != 0:
                        ell_map[ti, di] += (contact.minor / contact.major) * (1 - td) * (1 - dd)
                        ell_map[ti + 1, di] += (contact.minor / contact.major) * td * (1 - dd)
                        ell_map[ti, di + 1] += (contact.minor / contact.major) * (1 - td) * dd
                        ell_map[ti + 1, di + 1] += (contact.minor / contact.major) * td * dd

        #cv2.imshow('e', force_map * 0.05)
        #cv2.waitKey(0)
        feature.extend(force_map.flatten())
        feature.extend(area_map.flatten())
        feature.extend(inten_map.flatten())
        feature.extend(ell_map.flatten())
        return feature
    
    def getKeyContact(self, frame): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add a member variable 'feature'
        DELAY = 5 # in frames
        contacts = []

        for contact in frame.contacts:
            duration = len(self.frames) - self.start_frame[contact.id]
            if duration == DELAY or (duration < DELAY and contact.state == 3):
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

    pickle.dump([X, Y, Z], open('data.pickle', 'wb'))
    
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    '''
    clf = svm.SVC(gamma='auto', class_weight='balanced')
    print('Positive samples = %d' % (np.sum(Y == 1)))
    print('Negative samples = %d' % (np.sum(Y == 0)))
    print('Accuracy = %f' % (np.mean(cross_val_score(clf, X, Y, cv=5))))

    print('Total time = %f' % (time.perf_counter() - start_time))

    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
    '''

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    test_users = ['swn', 'plh', 'grc', 'hxz']
    for x, y, z in zip(X, Y, Z):
        if y != -1:
            if z in test_users:
                X_test.append(x)
                Y_test.append(y)
            else:
                X_train.append(x)
                Y_train.append(y)
    clf = svm.SVC(gamma='auto', class_weight='balanced')
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print(float(sum(np.array(Y_test) == np.array(Y_pred))) / len(Y_test))
