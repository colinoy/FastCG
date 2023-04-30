import logging

def singleton(cls):
    instances = {}
    def get_instance():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]
    return get_instance()

@singleton
class Logger():
    def __init__(self, verbose=0):
        self.logr = logging.getLogger('root')
        self.levels = [logging.WARNING, logging.INFO, logging.DEBUG]
        self.level = self.levels[min(verbose, len(self.levels) - 1)]
        print("Logger level: " + str(self.level))
        self.logr.setLevel(self.level)
        formatter = logging.Formatter('[%(asctime)s] %(filename)s:%(lineno)d (%(funcName)s) %(levelname)s - %(message)s','%m-%d %H:%M:%S')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.logr.addHandler(handler)

        self.info = self.logr.info
        self.debug = self.logr.debug
        self.warning = self.logr.warning
        self.error = self.logr.error
        self.critical = self.logr.critical
        self.exception = self.logr.exception
        self.log = self.logr.log

    def set_verbosity(self, verbosity):
        if type(verbosity) == int:
            self.level = self.levels[min(verbosity, len(self.levels) - 1)]
        elif verbosity in self.levels:
            self.level = verbosity
        print (f'Updated logger level to {self.level}')
        self.logr.setLevel(self.level)