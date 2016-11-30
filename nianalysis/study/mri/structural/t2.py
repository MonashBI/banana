from itertools import chain
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.data_formats import nifti_gz_format
from ..base import MRStudy


class T2Study(MRStudy):

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t2', nifti_gz_format),
        inherit_from=chain(MRStudy.generated_dataset_specs()))
