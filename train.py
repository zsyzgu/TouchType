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

class History():
    def __init__(self):
        N = 20
        self.contacts = [[] for i in range(N)]
        self.timestamps = []
        self.contacts_num = []
        self.total_forces = []
        self.total_areas = []

    def _getSequence(self, X):
        return [np.mean(X), np.std(X), np.min(X), np.max(X), stats.skew(X), stats.kurtosis(X)]

    def updateFrame(self, frame):
        self.timestamps.append(frame.timestamp)
        self.contacts_num.append(len(frame.contacts))
        total_force = 1
        total_area = 1
        for contact in frame.contacts:
            total_force += contact.force
            total_area += contact.area
            if contact.state == 1:
                self.contacts[contact.id] = []
            self.contacts[contact.id].append(contact)
        self.total_forces.append(total_force)
        self.total_areas.append(total_area)

    def getFeature(self, id):
        feature = []
        contacts = self.contacts[id]
        length = len(contacts)
        term = min(length, 5)

        for contact in contacts[:term]:
            if contact.major == 0:
                return []

        areas = [contact.area for contact in contacts[:term]]
        forces = [contact.force for contact in contacts[:term]]
        intens = [contact.force / contact.area for contact in contacts[:term]]
        frac_areas = [contacts[i].area / self.total_areas[-length + i] for i in range(term)]
        frac_forces = [contacts[i].force / self.total_forces[-length + i] for i in range(term)]
        ells = [float(contact.minor) / contact.major for contact in contacts[:term]]
        feature += self._getSequence(areas) + self._getSequence(forces) + self._getSequence(intens) + self._getSequence(ells) + self._getSequence(frac_areas) + self._getSequence(frac_forces)
        
        duration = self.timestamps[-1] - self.timestamps[-length]
        st_contacts_num = self.contacts_num[-length]
        en_contacts_num = self.contacts_num[-1]
        feature += [duration, st_contacts_num, en_contacts_num]

        return feature

def input(user, session, X, Y, Z):
    frames = compress_pickle.load('data/' + user + '/' + str(session) + '_labeled.gz')
    
    history = History()
    for frame in frames:
        history.updateFrame(frame)

        for contact in frame.contacts:
            if len(history.contacts[contact.id]) == 5 or contact.state == 3:
                feature = history.getFeature(contact.id)
                if len(feature) != 0:
                    X.append(feature)
                    Y.append(contact.label)
                    Z.append(user)

if __name__ == "__main__":
    file_name = DataManager(is_write=False).getFileName()
    tags = file_name.split('/')

    if tags[0] != 'xxx':
        users = [tags[0]]
    else:
        users = os.listdir('data/')

    if tags[1] != 'x':
        sessions = [int(tags[1])]
    else:
        sessions = [1, 2, 3, 4, 5]
    
    X = []
    Y = []
    Z = [] # users
    for user in users:
        for session in sessions:
            input(user, session, X, Y, Z)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
            
    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    clf = svm.SVC(gamma='auto')
    #clf = tree.DecisionTreeClassifier(criterion='entropy')
    # TODO: balance positive/negative samples
    print('Positive samples = %d' % (np.sum(Y == 1)))
    print('Negative samples = %d' % (np.sum(Y == 0)))
    print('Accuracy = %f' % (np.mean(cross_val_score(clf, X, Y, cv=5))))

    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
