from enum import Enum


class StrEnum(str, Enum):
    """ Define a string enum class """
    pass


class RunnerPhase(StrEnum):
    """ Model runner phase enum  """
    TRAIN = 'train'
    VALIDATE = 'validate'
    PREDICT = 'predict'


class LossFunction(StrEnum):
    """ Loss function for neural TPP model  """
    LOGLIKE = 'loglike'
    PARTIAL_TIME_LOSS = 'rmse'
    PARTIAL_EVENT_LOSS = 'accuracy'


class LogConst:
    DEFAULT_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s'


class PredOutputIndex:
    TimePredIndex = 0
    TypePredIndex = 1
