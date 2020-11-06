import pickle
from frame_data import FrameData
from frame_data import ContactData
from sklearn import svm
import numpy as np
import random
from sklearn import tree
from sklearn.model_selection import cross_val_score
from scipy import stats
from sklearn.preprocessing import StandardScaler
import cv2

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
        
        #duration = self.timestamps[-1] - self.timestamps[-length]
        #st_contacts_num = self.contacts_num[-length]
        #en_contacts_num = self.contacts_num[-1]
        #feature += [duration, st_contacts_num, en_contacts_num]

        return feature

def process(X, Y, frames, label):
    history = History()

    for frame in frames:
        history.updateFrame(frame)

        for contact in frame.contacts:
            if len(history.contacts[contact.id]) == 5 or contact.state == 3:
                feature = history.getFeature(contact.id)
                if len(feature) != 0:
                    X.append(feature)
                    Y.append(label)

def input(files):
    frames = []
    for file in files:
        frames.extend(pickle.load(open('data/' + file + '.pickle', 'rb')))
    return frames

if __name__ == "__main__":
    users = 6
    N_frames = input(['N_' + str(i) for i in range(1, users + 1)])
    P_frames = input(['P_' + str(i) for i in range(1, users + 1)])

    X = []
    Y = []
    process(X, Y, N_frames, 0)
    process(X, Y, P_frames, 1)

    scalar = StandardScaler()
    scalar.fit(X)
    X = scalar.transform(X)

    clf = svm.SVC(gamma='auto')
    #clf = tree.DecisionTreeClassifier(criterion='entropy')
    print(len(Y))
    print(np.mean(cross_val_score(clf, X, Y, cv=5)))
    clf.fit(X, Y)
    pickle.dump([scalar, clf], open('model.pickle', 'wb'))
