from datetime import datetime

import logging
import os
import sys
import uuid

def generate_logger(name=None):
    if name == None:
        # setup the path to the logfile
        from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        logname = f'log_{from_date}.log'
        
        # folder = './log/logs/' + str(uuid.uuid4()) + '/'
        # os.mkdir(folder)
        folder = './log/logs/'
        
        logfile = os.path.join(folder, logname)
    else:
        logfile = name

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(filename=logfile)
    stdout_handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s - in file: %(filename)s - line: %(lineno)d", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    
    return logger