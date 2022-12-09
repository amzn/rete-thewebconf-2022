import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'            # for handling CTRL-C on Windows

# some constants
TRAIN = 0
VALID = 1
TEST  = 2

MODE2STR = {TRAIN: 'train', VALID: 'valid', TEST: 'test'}
STR2MODE = {'train': TRAIN, 'valid': VALID, 'test': TEST}