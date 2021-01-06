import numpy
from pynput.keyboard import Key, Controller

class Layout():
    def __init__(self):
        lines = open('model/layout.txt', 'r').readlines()

        tags0 = lines[0].split()
        H, W = float(tags0[0]), float(tags0[1])

        self.mapping = []

        for line in lines[1:]:
            tags = line.split()
            y, x0, x_step, letters = float(tags[0]), float(tags[1]), float(tags[2]), tags[3]
            x = x0
            for ch in letters:
                self.mapping.append((x/W, y/H, ch))
                x += x_step
        
        for i in range(len(self.mapping)):
            x, y, ch = self.mapping[i]

            if ch in 'qwertyuiopasdfghjklzxcvbnm`1234567890-=[];,.':
                key = ch
            if ch == 'B':
                key = Key.backspace
            if ch == 'S':
                key = Key.shift
            if ch == 'E':
                key = Key.enter
            if ch == '_':
                key = Key.space
            if ch == 'T':
                key = Key.tab
            if ch == 'C':
                key = Key.ctrl
            if ch == 'A':
                key = Key.alt
            if ch == 'M':
                key = Key.cmd
            if ch == 'U':
                key = Key.up
            if ch == 'D':
                key = Key.down
            if ch == 'L':
                key = Key.left
            if ch == 'R':
                key = Key.right
            if ch == '<':
                key = Key.esc
            
            self.mapping[i] = (x, y, key)

    def decode(self, x0, y0):
        min_dist2 = 1e6
        key = None

        for (x,y,k) in self.mapping:
            dist2 = (x-x0)**2+(y-y0)**2
            if dist2 < min_dist2:
                min_dist2 = dist2
                key = k

        return key
