from arcana.study.base import StudyMetaClass
from .base import MriStudy
from arcana.study import ParameterSpec
from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.utils.interfaces import DummyReconAll as ReconAll
from banana.requirement import freesurfer_req, ants_req, fsl_req
from banana.citation import freesurfer_cites, fsl_cite
from nipype.interfaces import fsl, ants
from arcana.utils.interfaces import Merge
from banana.file_format import (
    nifti_gz_format, text_matrix_format)
from arcana.data import FilesetSpec
from arcana.utils.interfaces import JoinPath
from .base import MriStudy
from arcana.study.base import StudyMetaClass
from arcana.study import ParameterSpec


class T2Study(MriStudy, metaclass=StudyMetaClass):

    add_param_specs = [
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.5),
        ParameterSpec('bet_reduce_bias', False)]
