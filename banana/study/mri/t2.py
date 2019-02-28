from arcana.study.base import StudyMetaClass
from .base import MriStudy
from arcana.study import ParameterSpec


class T2Study(MriStudy, metaclass=StudyMetaClass):

    add_param_specs = [
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.5),
        ParameterSpec('bet_reduce_bias', False)]
