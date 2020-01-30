import sys
import math
import time
import tqdm


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None


    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1):
        # if '\r' in message: is_file=0

        if is_terminal == 1:
            # print(self.terminal)
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)
        # print(self.file)
        if self.file is not None:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
