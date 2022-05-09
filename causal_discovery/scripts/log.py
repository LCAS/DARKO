import logging
import os

from constants import *

class Log:
    def __init__(self, folder, filename, name, level):
        """
        Log contructor

        Args:
            folder (str): log folder
            filename (str): log filename
            name (str): logger name
        """
        self.folder = folder
        self.filename = filename
        self.name = name
        
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # create console handler and set level to debug
        ch = logging.FileHandler(filename = folder + '/' + filename)
        ch.setLevel(level)

        # create formatter
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        ch.setFormatter(self.formatter)

        # add ch to logger
        self.logger.addHandler(ch)
        
        
    def debug(self, msg):
        """
        Write DEBUG msg

        Args:
            msg (str): DEBUG message
        """
        self.logger.debug(msg)

    
    def info(self, msg):
        """
        Write INFO msg

        Args:
            msg (str): INFO message
        """
        self.logger.info(msg)
        
    
    def warn(self, msg):
        """
        Write WARN msg

        Args:
            msg (str): WARN message
        """
        self.logger.warn(msg)
        
        
    def error(self, msg):
        """
        Write ERROR msg

        Args:
            msg (str): ERROR message
        """
        self.logger.error(msg)
    

log = Log(LOG_FOLDER, LOG_FILENAME, NODE_NAME, LOG_LEVEL)