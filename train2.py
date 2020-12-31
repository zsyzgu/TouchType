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
import pandas as pd

class History():
    def __init__(self):
        self.N = 16
        self.frames = []
        self.start_frame = np.zeros(self.N)
        self.total_forces = []
        self.total_areas = []
        self.total_intens = []

    def _getSequence(self, X):
        # a Faster Implementation of: return [np.mean(X), np.std(X), np.min(X), np.max(X), stats.skew(X), stats.kurtosis(X)]
        niu = np.mean(X)
        std = np.std(X)
        minX = np.min(X)
        maxX = np.max(X)
        niu2 = np.mean([x**2 for x in X])
        sigma = max(0, niu2 - niu**2) ** 0.5
        if sigma == 0:
            skew = 0
            kurt = 0
        else:
            niu3 = np.mean([x**3 for x in X])
            niu4 = np.mean([(x-niu)**4 for x in X])
            skew = skew =(niu3-3*niu*sigma**2-niu**3)/(sigma**3)
            kurt=niu4/(sigma**4)-3
        return [niu, std, minX, maxX, skew, kurt]

    def updateFrame(self, frame):
        self.frames.append(frame)

        total_force = 0
        total_area = 0
        total_inten = 0
        for contact in frame.contacts:
            if contact.state == 1:
                self.start_frame[contact.id] = len(self.frames)
            total_force += contact.force
            total_area += contact.area
            total_inten += contact.force / contact.area
        
        self.total_forces.append(total_force)
        self.total_areas.append(total_area)
        self.total_intens.append(total_inten)

    def getContact(self, t, id):
        st = t
        en = t - 1
        while (st - 1 >= 0):
            key_contact = None
            for contact in self.frames[st - 1].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            if key_contact == None:
                break
            st -= 1
            if key_contact.state == 1:
                break
        while (en + 1 < len(self.frames)):
            key_contact = None
            for contact in self.frames[en + 1].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            if key_contact == None:
                break
            en += 1
            if key_contact.state == 3:
                break
        contacts = []
        for t in range(st, en + 1):
            key_contact = None
            for contact in self.frames[t].contacts:
                if contact.id == id:
                    key_contact = contact
                    break
            contacts.append(key_contact)
        return st, en, contacts

    def completeTaps(self, taps, length):
        no_tap = [0, 0, 0, 0, 0, 0]
        if len(taps) > length:
            taps = taps[:length]
        while len(taps) < length:
            taps.append(no_tap)
        return taps

    def getFeature(self, id):
        st, en, contacts = self.getContact(len(self.frames)-1, id)

        for contact in contacts:
            if contact.major == 0:
                return []

        feature = []

        areas = [contact.area for contact in contacts]
        forces = [contact.force for contact in contacts]
        intens = [contact.force / contact.area for contact in contacts]
        ells = [float(contact.minor) / contact.major for contact in contacts]
        feature += self._getSequence(areas) + self._getSequence(forces) + self._getSequence(intens) + self._getSequence(ells)
        
        dist2edge = [min(min(contact.x, 1-contact.x),1-contact.y) for contact in contacts]
        dist2click = [((contacts[i].x-contacts[0].x)**2+(contacts[i].y-contacts[0].y)**2)**0.5 for i in range(len(contacts))]
        dist2corner = [min((contact.x-0)**2+(contact.y-1)**2, (contact.x-1)**2+(contact.y-1)**2)**0.5 for contact in contacts]
        feature += self._getSequence(dist2edge) + self._getSequence(dist2corner) + self._getSequence(dist2click)

        frac_areas = [contacts[i].area / self.total_areas[st+i] for i in range(len(contacts))]
        frac_forces = [contacts[i].force / self.total_forces[st+i] for i in range(len(contacts))]
        frac_intens = [(contacts[i].force / contacts[i].area) / self.total_intens[st+i] for i in range(len(contacts))]
        feature += self._getSequence(frac_areas) + self._getSequence(frac_forces) + self._getSequence(frac_intens)

        other_taps = []
        for t in range(en-1, max(st-int(5*Board.FPS),0)-1, -1):
            for contact in self.frames[t].contacts:
                if contact.state == 1 and not (t == st and contact.id == id) and len(other_taps) < 10: # whether the start of another contact
                    S, E, C = self.getContact(t, contact.id)
                    st_time = float(S - st) / 50
                    en_time = float(E - st) / 50
                    dist = np.mean([((c.x-contacts[-1].x)**2+(c.y-contacts[-1].y)**2)**0.5 for c in C])
                    force = np.mean([c.force for c in C])/np.mean(forces)
                    area = np.mean([c.area for c in C])/np.mean(areas)
                    inten = np.mean([c.force / c.area for c in C])/np.mean(ells)
                    other_taps.append([st_time, en_time, dist, force, area, inten])
        
        pre_taps = [tap for tap in other_taps if tap[0] < 0] # Add 5 pre taps
        pre_taps.reverse()
        feature.extend(np.array(self.completeTaps(pre_taps, 5)).flatten())
        post_taps = [tap for tap in other_taps if tap[0] > 0] # Add 2 post taps
        feature.extend(np.array(self.completeTaps(post_taps, 2)).flatten())
        close_taps = [tap for tap in other_taps if tap[1] >= 0] # Add 3 closest taps during this tap
        close_taps.sort(key=lambda tap: tap[2])
        feature.extend(np.array(self.completeTaps(close_taps, 2)).flatten())

        return feature
    
    def getKeyContact(self, frame): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add member variables 'feature', 'duration'
        D1, D2 = 3, 5 # in frames
        contacts = []

        for contact in frame.contacts:
            duration = len(self.frames) - self.start_frame[contact.id]
            if (D1 <= duration and duration <= D2) or (duration < D1 and contact.state == 3):
                feature = self.getFeature(contact.id)
                if len(feature) != 0:
                    contact.duration = duration
                    contact.feature = feature
                    contacts.append(contact)

        return contacts

def input(user, session):
    X = []
    Y = []
    Z = []

    frames = pickle.load(open('data/' + user + '/' + str(session) + '.simple', 'rb')) # without force_array

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
    
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    users = os.listdir('data/')
    accs = []
    for test_id in range(len(users)):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        for x, y, z in zip(X, Y, Z):
            if y != -1:
                if z == users[test_id]:
                    X_test.append(x)
                    Y_test.append(y)
                else:
                    X_train.append(x)
                    Y_train.append(y)
        clf = svm.SVC(gamma='auto', class_weight='balanced')
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_test)
        acc = float(sum(np.array(Y_test) == np.array(Y_pred))) / len(Y_test)
        print(users[test_id], acc)
        accs.append(acc)
    print('Total Acc =', np.mean(accs), np.std(accs))

    clf = svm.SVC(gamma='auto', class_weight='balanced')
    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
