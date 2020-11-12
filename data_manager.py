import sys
import os

class DataManager():
    def __init__(self, is_write = True):
        self.is_write = is_write

    def _print_usage(self):
        print('[Usage] python script.py folder/index')

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
        folders = os.listdir(root)
        if folder not in folders:
            os.mkdir(root + folder)
        
        files = os.listdir(root + folder)
        pres = [file.split('.')[0] for file in files]
        if (index in pres) and self.is_write:
            print('\033[1;31;40m[WARNING]\033[0m Folder exists. Still continue? [y/n]')
            answer = input()
            if answer != 'y':
                exit()

        file_name = folder + '/' + index
        return file_name

if __name__ == "__main__":
    file_name = DataManager().getFileName()
