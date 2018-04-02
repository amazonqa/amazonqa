
import argparse
import constants as C
from datetime import datetime

class Logger:

    def __init__(self, logfilename = None, clear_file=True):
        self.logfilename = logfilename if logfilename else _log_file_name()
        self.log('', clear=clear_file)

    def log(self, line, clear=False):
        assert clear in [True, False]
        print(line)
        with open(self.logfilename, 'w' if clear else 'a') as fp:
            fp.write('%s\n' % line)

def _log_file_name():
    parser = argparse.ArgumentParser()
    default_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    parser.add_argument('--%s' % C.LOG_FILENAME, dest=C.LOG_FILENAME, type=str, default='%s.log' % default_name)
    args, _ = parser.parse_known_args()
    return vars(args)[C.LOG_FILENAME]
