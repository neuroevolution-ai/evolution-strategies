from enum import Enum


class ConfigValues(Enum):
    OPTIMIZER_ADAM = 'adam'
    OPTIMIZER_SGD = 'sgd'
    RETURN_PROC_MODE_CR = 'centered_rank'
    RETURN_PROC_MODE_SIGN = 'sign'
    RETURN_PROC_MODE_CR_SIGN = 'centered_sign_rank'
