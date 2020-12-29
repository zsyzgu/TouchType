import pickle
import numpy as np

def check(a):
    accs = []
    for i in range(len(a)):
        accs.append(sum(a[i])/len(a[i]))
    print(np.mean(accs), np.std(accs))

a1 = pickle.load(open('1.pickle', 'rb'))
a2 = pickle.load(open('2.pickle', 'rb'))
a3 = pickle.load(open('3.pickle', 'rb'))

check(a1)
check(a2)
check(a3)

accs = []
for i in range(len(a1)):
    correct = 0
    for j in range(len(a1[i])):
        if (a1[i][j] and a2[i][j]) or (a1[i][j] and a3[i][j]) or (a2[i][j] and a3[i][j]):
            correct += 1
    accs.append(float(correct) / len(a1[i]))
print(np.mean(accs), np.std(accs))

