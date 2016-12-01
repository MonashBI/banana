from itertools import chain
from nianalysis.data_formats import (
    nifti_gz_format, freesurfer_format, text_matrix_format)
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from ...combined import CombinedStudy
from ..coregistered import CoregisteredStudy
from .t1 import T1Study
from .t2 import T2Study


class T1T2Study(CombinedStudy):
    """
    T1 and T2 weighted MR dataset, with the T2-weighted coregistered to the T1.
    """

    def freesurfer_pipeline(self, **kwargs):
        pipeline = self.TranslatedPipeline(
            'freesurfer', self.t1_study.freesurfer_pipeline(**kwargs), self,
            add_inputs=['t2_coreg'])
        recon_all = pipeline.node('recon_all')
        print recon_all.inputs
        recon_all.inputs.use_T2 = True
        # Connect T2-weighted input
        pipeline.connect_input('t2_coreg', recon_all, 'T2_file')
        pipeline.assert_connected()
        return pipeline

    registration_pipeline = CombinedStudy.translate(
        'coreg_study', CoregisteredStudy.registration_pipeline)

    sub_study_specs = {'t1_study': (T1Study,
                                    {'t1': 't1',
                                     'freesurfer': 'freesurfer'}),
                       't2_study': (T2Study, {'t2_coreg': 't2'}),
                       'coreg_study': (CoregisteredStudy,
                                       {'t1': 'reference',
                                        't2': 'to_register',
                                        't2_coreg': 'registered',
                                        'coreg_matrix': 'matrix'})}

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('t2', nifti_gz_format),
        DatasetSpec('t2_coreg', nifti_gz_format, registration_pipeline),
        DatasetSpec('coreg_matrix', text_matrix_format, registration_pipeline),
        DatasetSpec('freesurfer', freesurfer_format, freesurfer_pipeline),
        inherit_from=chain(T1Study.generated_dataset_specs(),
                           T2Study.generated_dataset_specs()))
