
import argparse
import constants as C
from datetime import datetime
import os

class Logger:

    def __init__(self, logfilename = None, clear_file=True, base_dir=C.LOG_DIR):
        self.logfilename = logfilename if logfilename else _log_file_name(base_dir)
        self.log('', clear=clear_file)

    def log(self, line, clear=False):
        assert clear in [True, False]
        print(line)
        with open(self.logfilename, 'w' if clear else 'a') as fp:
            fp.write('%s\n' % line)

def _log_file_name(base_dir):
    parser = argparse.ArgumentParser()
    default_name = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    _ensure_path(base_dir)
    parser.add_argument('--%s' % C.LOG_FILENAME, dest=C.LOG_FILENAME, type=str, default='%s/%s.log' % (base_dir, default_name))
    args, _ = parser.parse_known_args()
    return vars(args)[C.LOG_FILENAME]

def _ensure_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
