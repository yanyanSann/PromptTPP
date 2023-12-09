import numpy as np
import logging
from neural_tpp.utils import LogConst


class LogHandler:
    def __init__(self,
                 log_dir,
                 console_level=logging.INFO,
                 file_level=logging.INFO):
        self.logger = logging.getLogger(log_dir)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(LogConst.DEFAULT_FORMAT)

        for handler in self.logger.handlers:
            handler.close()
        self.logger.handlers[:] = []

        fh = logging.FileHandler(log_dir)
        fh.setFormatter(fmt)
        fh.setLevel(file_level)
        self.logger.addHandler(fh)

        if console_level is not None:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel(console_level)
            self.logger.addHandler(sh)
        self.logger.propagate = False

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)

    def init(self, args):
        self.logger.info('Configuration initialized')
        for argname in args:
            self.logger.info("{} : {}".format(argname, args[argname]))

        self.current_best = {
            'loglike': np.finfo(float).min,
            'loss': np.finfo(float).max,
            'rmse': np.finfo(float).max,
            'acc': np.finfo(float).min,
        }
        self.episode_best = 'NeverUpdated'

    def update_best(self, key, value, episode):
        updated = False
        if key == 'loglike' or key == 'acc':
            if value > self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        elif key == 'loss' or key == 'rmse':
            if value < self.current_best[key]:
                updated = True
                self.current_best[key] = value
                self.episode_best = episode
        else:
            raise Exception("unknown key {}".format(key))
        return updated
