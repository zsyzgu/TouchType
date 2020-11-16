import sys
import os

class DataManager():
    def __init__(self, is_write = True):
        self.is_write = is_write

    def _print_usage(self):
        print('[Usage] python script.py folder/index')

    def _file_exist_warning(self):
        print('\033[1;31;40m[WARNING]\033[0m File exists. Still continue? [y/n]')
        answer = input()
        if len(answer) != 1:
            self._file_exist_warning()
        elif answer != 'y':
            exit()

    def judgeFileExistance(self, file_path):
        if os.path.exists(file_path):
            self._file_exist_warning()

    def getFileName(self):
        if len(sys.argv) != 2:
            self._print_usage()
            exit()
        tags = sys.argv[1].split('/')
        if len(tags) != 2 or len(tags[0]) == 0 or len(tags[1]) == 0:
            self._print_usage()
            exit()
        folder = tags[0]
        index = tags[1]

        root = 'data/'
        if self.is_write:
            folders = os.listdir(root)
            if folder not in folders:
                os.mkdir(root + folder)
            
            files = os.listdir(root + folder)
            pres = [file.split('.')[0] for file in files]
            if index in pres:
                self._file_exist_warning()

        file_name = folder + '/' + index
        return file_name

