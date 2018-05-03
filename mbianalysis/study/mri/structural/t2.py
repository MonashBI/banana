from nianalysis.study.base import StudyMetaClass
from ..base import MRIStudy
from nianalysis.option import OptionSpec


class T2Study(MRIStudy):

    __metaclass__ = StudyMetaClass

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.5),
        OptionSpec('bet_reduce_bias', False)]
