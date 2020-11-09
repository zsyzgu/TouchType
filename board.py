import sys
import cv2
import time
import numpy as np
import pickle
import keyboard
from sklearn import svm
sys.path.append('./sensel-lib-wrappers/sensel-lib-python')
import sensel
from frame_data import FrameData
from frame_data import ContactData
from history import History
import _thread

class Board():
    MAX_X = 230.0
    MAX_Y = 130.0
    FPS = 50

    def __init__(self):
        self._openSensel()
        self._initFrame()

    def __del__(self):
        self._closeSensel()

    def _openSensel(self):
        handle = None
        (error, device_list) = sensel.getDeviceList()
        if device_list.num_devices != 0:
            (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
        self.handle = handle

    def _initFrame(self):
        (error, self.info) = sensel.getSensorInfo(self.handle)
        error = sensel.setFrameContent(self.handle, 0x0F)
        error = sensel.setContactsMask(self.handle, 0x0F)
        (error, frame) = sensel.allocateFrameData(self.handle)
        error = sensel.startScanning(self.handle)
        self._frame = frame
        self.frames = []
        try:
            _thread.start_new_thread(self._updateFrame, ())
        except:
            print("Thread Error")

    def _closeSensel(self):
        self.is_running = False
        error = sensel.freeFrameData(self.handle, self._frame)
        error = sensel.stopScanning(self.handle)
        error = sensel.close(self.handle)
    
    def _sync(self):
        if len(self.frames) > 0:
            while ((time.clock() - self.frames[-1].timestamp) * Board.FPS < 1):
                pass

    def _updateFrame(self):
        self.is_running = True
        while (self.is_running):
            error = sensel.readSensor(self.handle)
            (error, num_frames) = sensel.getNumAvailableFrames(self.handle)
            for i in range(num_frames):
                self._sync()
                timestamp = time.clock()
                error = sensel.getFrame(self.handle, self._frame)
            R = self.info.num_rows
            C = self.info.num_cols
            force_array = np.zeros((R, C))
            for r in range(R):
                force_array[r, :] = self._frame.force_array[r * C : (r + 1) * C]
            force_array *= 0.2
            frame = FrameData(force_array, timestamp)

            for i in range(self._frame.n_contacts):
                c = self._frame.contacts[i]
                x = c.x_pos / Board.MAX_X
                y = c.y_pos / Board.MAX_Y
                contact = ContactData(c.id, c.state, x, y, c.area, c.total_force, c.major_axis, c.minor_axis, c.delta_x, c.delta_y, c.delta_force, c.delta_area)
                frame.append_contact(contact)
            self.frames.append(frame)

    def getFrame(self):
        while (len(self.frames) == 0):
            time.sleep(0.001)
        return self.frames[-1]
    
    def get_frames_within(self, start_time, end_time):
        frames = []
        for frame in self.frames:
            if start_time <= frame.timestamp and frame.timestamp <= end_time:
                frames.append(frame)
        return frames

    def getFrameTime(self):
        if len(self.frames) >= 2:
            return round(self.frames[-1].timestamp - self.frames[-2].timestamp, 5)
        return 0
