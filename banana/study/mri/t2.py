from arcana.study.base import StudyMetaClass
from .base import MriStudy
from arcana.study import ParamSpec


class T2Study(MriStudy, metaclass=StudyMetaClass):

    add_param_specs = [
        ParamSpec('bet_robust', True),
        ParamSpec('bet_f_threshold', 0.5),
        ParamSpec('bet_reduce_bias', False)]
