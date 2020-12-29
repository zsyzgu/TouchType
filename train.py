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

        '''
        frac_forces = []
        frac_areas = []
        frac_intens = []
        for i in range(length):
            total_force = 0
            total_area = 0
            total_inten = 0
            curr_contact = contacts[-length + i]
            for point in range(self.N):
                other_contacts = self.contacts[point]
                if length - i <= len(other_contacts):
                    other_contact = other_contacts[-length + i]
                    total_force += other_contact.force
                    total_area += other_contact.area
                    total_inten += other_contact.force / other_contact.area
            frac_forces.append(curr_contact.force / total_force)
            frac_areas.append(curr_contact.area / total_area)
            frac_intens.append((curr_contact.force / curr_contact.area) / total_inten)
        feature += self._getSequence(frac_forces) + self._getSequence(frac_areas) + self._getSequence(frac_intens)
        '''

        areas = [contact.area for contact in contacts]
        forces = [contact.force for contact in contacts]
        intens = [contact.force / contact.area for contact in contacts]
        ells = [float(contact.minor) / contact.major for contact in contacts]
        feature += self._getSequence(areas) + self._getSequence(forces) + self._getSequence(intens) + self._getSequence(ells)

        dist2edge = [min(min(contact.x, 1-contact.x),min(contact.y, 1-contact.y)) for contact in contacts]
        feature += self._getSequence(dist2edge)

        dist2click = [((contacts[i].x-contacts[0].x)**2+(contacts[i].y-contacts[0].y)**2)**0.5 for i in range(len(contacts))]
        feature += self._getSequence(dist2click)

        return feature
    
    def getKeyContact(self, frame): # Return contacts which are right to judge (5 frames or the end).
        # Each 'contact' add a member variable 'feature'
        DELAY_L = 5 # in frames
        DELAY_R = 10
        contacts = []

        for contact in frame.contacts:
            duration = len(self.contacts[contact.id])
            #if duration == DELAY or (duration < DELAY and contact.state == 3):
            #if (DELAY_L <= duration and duration <= DELAY_R) or (duration < DEAY_L and contact.state == 3):
            if contact.state == 3:
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
    
    '''
    clf = svm.SVC(gamma='auto', class_weight='balanced')
    print('Positive samples = %d' % (np.sum(Y == 1)))
    print('Negative samples = %d' % (np.sum(Y == 0)))
    print('Accuracy = %f' % (np.mean(cross_val_score(clf, X, Y, cv=5))))

    print('Total time = %f' % (time.perf_counter() - start_time))

    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
    '''
