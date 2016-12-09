from itertools import chain
from nianalysis.study.base import set_dataset_specs
from ..base import MRStudy


class T2Study(MRStudy):

    def brain_mask_pipeline(self, robust=True, threshold=0.5,
                            reduce_bias=False, **kwargs):
        return super(T2Study, self).brain_mask_pipeline(
            robust=robust, threshold=threshold, reduce_bias=reduce_bias,
            **kwargs)

    _dataset_specs = set_dataset_specs(
        inherit_from=chain(MRStudy.dataset_specs()))
