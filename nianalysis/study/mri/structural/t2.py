from arcana.study.base import StudyMetaClass
from ..base import MRIStudy
from arcana.option import ParameterSpec


class T2Study(MRIStudy, metaclass=StudyMetaClass):

    add_option_specs = [
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.5),
        ParameterSpec('bet_reduce_bias', False)]
