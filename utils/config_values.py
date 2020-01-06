from enum import Enum


class ConfigValues(Enum):
    OPTIMIZER_ADAM = "adam"
    OPTIMIZER_SGD = "sgd"
    RETURN_PROC_MODE_CR = "centered_rank"
    RETURN_PROC_MODE_SIGN = "sign"
    RETURN_PROC_MODE_CR_SIGN = "centered_sign_rank"


class LogColumnHeaders(Enum):
    GEN = "Generation"
    GEN_REW_MEAN = "GenRewMean"
    GEN_REW_STD = "GenRewStd"
    GEN_LEN_MEAN = "GenLenMean"
    EVAL_GEN_REW_MEAN = "EvalGenRewardMean"
    EVAL_GEN_REW_STD = "EvalGenRewardStd"
    EVAL_GEN_LEN_MEAN = "EvalGenLengthMean"
    EVAL_GEN_COUNT = "EvalGenCount"
    EPS_THIS_GEN = "EpisodesThisGen"
    EPS_SO_FAR = "EpisodesSoFar"
    TIMESTEPS_THIS_GEN = "TimestepsThisGen"
    TIMESTEPS_SO_FAR = "TimestepsSoFar"
    UNIQUE_WORKERS = "UniqueWorkers"
    RESULTS_SKIPPED_FRAC = "ResultsSkippedFrac"
    OBS_COUNT = "ObCount"
    TIME_ELAPSED_THIS_GEN = "TimeElapsedThisGen"
    TIME_ELAPSED = "TimeElapsed"
    TIME_PREDICT_MIN = "TimePredictMin"
    TIME_PREDICT_MAX = "TimePredictMax"
    TIME_PREDICT_MEAN = "TimePredictMean"
    TIME_PREDICT_COUNT = "TimePredictCount"


class EvaluationColumnHeaders(Enum):
    GEN = "Generation"
    EVAL_COUNT_PER_GEN = "Eval_per_Gen"
    EVAL_REW_MEAN = "Eval_Rew_Mean"
    EVAL_REW_STD = "Eval_Rew_Std"
    EVAL_LEN_MEAN = "Eval_Len_Mean"
    REW_PREFIX = "Rew_"
    LEN_PREFIX = "Len_"
