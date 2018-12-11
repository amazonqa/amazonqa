import argparse
from datetime import datetime
import os

LOG_FILENAME = 'log_filename'

class Logger:

    def __init__(self, logfilename = None, clear_file=True, base_dir='logs', verbose = True):
        self.logfilename = logfilename if logfilename else _log_file_name(base_dir)
        self.verbose = verbose
        if self.verbose:
            self.log('', clear=clear_file)

    def log(self, line, clear=False):
        assert clear in [True, False]
        if self.verbose:
            print(line)
            with open(self.logfilename, 'w' if clear else 'a') as fp:
                fp.write('%s\n' % line)

def _log_file_name(base_dir):
    parser = argparse.ArgumentParser()
    default_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    _ensure_path(base_dir)
    parser.add_argument('--%s' % LOG_FILENAME, dest=LOG_FILENAME, type=str, default='%s.log' % default_name)
    args, _ = parser.parse_known_args()
    filename = vars(args)[LOG_FILENAME]
    return '%s/%s' % (base_dir, filename)

def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
