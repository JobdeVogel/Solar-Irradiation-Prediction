from datetime import datetime

import logging
import os
import sys
import uuid

def generate_logger(identifier=None, stdout=True):
    abs_path = os.path.dirname(os.path.dirname(__file__))

    if identifier != None:
        # setup the path to the logfile
        from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        logname = f'log_{from_date}_ID_{identifier}.log'
        
        folder = os.path.join(abs_path, 'log\\logs')
        
        logfile = os.path.join(folder, logname)
    else:
        # setup the path to the logfile
        from_date = "{:%Y_%m_%d_%H_%M_%S}".format(datetime.now())
        logname = f'log_{from_date}.log'
        
        folder = os.path.join(abs_path, 'log\\logs')
        
        logfile = os.path.join(folder, logname)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(filename=logfile)
    
    if stdout:
        stdout_handler = logging.StreamHandler(stream=sys.stdout)

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s - in file: %(filename)s - line: %(lineno)d", datefmt='%Y-%m-%d %H:%M:%S')

    file_handler.setFormatter(formatter)
    if stdout:
        stdout_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    if stdout:
        logger.addHandler(stdout_handler)
    
    return logger