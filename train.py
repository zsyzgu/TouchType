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
        self.N = 16
        self.contacts = [[] for i in range(self.N)]
        self.timestamps = []

    def _getSequence(self, X):
        return [np.mean(X), np.std(X), np.min(X), np.max(X), stats.skew(X), stats.kurtosis(X)]
        #return [np.mean(X), np.min(X), np.max(X)]

    def updateFrame(self, frame):
        self.timestamps.append(frame.timestamp)
        flag = np.zeros(self.N)

        for contact in frame.contacts:
            flag[contact.id] = 1
            if contact.state == 1:
                self.contacts[contact.id] = []
            self.contacts[contact.id].append(contact)

        for i in range(self.N):
            if flag[i] == 0:
                self.contacts[i] = []

    def getFeature(self, id):
        feature = []
        contacts = self.contacts[id]
        length = len(contacts)

        for contact in contacts:
            if contact.major == 0:
                return []

        frac_forces = []
        frac_areas = []
        frac_delta_forces = []
        frac_delta_areas = []
        for i in range(length):
            total_force = 0
            total_area = 0
            total_delta_force = 1
            total_delta_area = 1
            curr_contact = contacts[-length + i]
            if i >= 1:
                last_contact = contacts[-length + i - 1]
            for point in range(self.N):
                other_contacts = self.contacts[point]
                if length - i <= len(other_contacts):
                    other_contact = other_contacts[-length + i]
                    total_force += other_contact.force
                    total_area += other_contact.area
                    if i >= 1 and (length - (i-1) <= len(other_contacts)):
                        other_last_contact = other_contacts[-length + i - 1]
                        total_delta_force += max(0, other_contact.force - other_last_contact.force)
                        total_delta_area += max(0, other_contact.area - other_last_contact.area)
            frac_forces.append(curr_contact.force / total_force)
            frac_areas.append(curr_contact.area / total_area)
            if i >= 1:
                frac_delta_forces.append((max(0, curr_contact.force - last_contact.force)) / total_delta_force)
                frac_delta_areas.append((max(0, curr_contact.area - last_contact.area)) / total_delta_area)
        feature += self._getSequence(frac_forces) + self._getSequence(frac_areas)
        feature += self._getSequence(frac_delta_forces) + self._getSequence(frac_delta_areas)

        areas = [contact.area for contact in contacts]
        forces = [contact.force for contact in contacts]
        intens = [contact.force / contact.area for contact in contacts]
        ells = [float(contact.minor) / contact.major for contact in contacts]
        feature += self._getSequence(areas) + self._getSequence(forces) + self._getSequence(intens) + self._getSequence(ells)

        delta_forces = [contacts[i].force - contacts[i-1].force for i in range(1, len(contacts))]
        delta_areas = [contacts[i].area - contacts[i-1].area for i in range(1, len(contacts))]
        feature += self._getSequence(delta_forces) + self._getSequence(delta_areas)

        xs = [contact.x for contact in contacts]
        ys = [contact.y for contact in contacts]
        feature += self._getSequence(xs) + self._getSequence(ys)

        dxs = [contacts[i].x - contacts[0].x for i in range(len(contacts))]
        dys = [contacts[i].y - contacts[0].y for i in range(len(contacts))]
        feature += self._getSequence(dxs) + self._getSequence(dys)

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

    frames = pickle.load(open('data/' + user + '/' + str(session) + '.simple', 'rb')) # without force_array
    #frames = compress_pickle.load('data/' + user + '/' + str(session) + '_checked.gz') # with force_array

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
