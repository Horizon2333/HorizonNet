import sys

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if(__name__ == "__main__"):
    sys.stdout = Logger(stream=sys.stdout)

    # now it works
    print('print something')
    print("output")