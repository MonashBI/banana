import os
from itertools import chain
from copy import copy
from nipype.pipeline import engine as pe
from nipype.interfaces.freesurfer.preprocess import ReconAll
from nipype.interfaces.spm import Info
from nianalysis.requirements import spm12_req, freesurfer_req
from nianalysis.citations import spm_cite, freesurfer_cites
from nianalysis.data_formats import (
    nifti_gz_format, nifti_format, freesurfer_format, text_matrix_format)
from nipype.interfaces.utility import Merge, Split
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nianalysis.interfaces.spm import MultiChannelSegment
from ..base import MRStudy
from ...combined import CombinedStudy
from ..coregistered import CoregisteredStudy


class T1Study(MRStudy):

    def segmentation_pipeline(self, seg_tool='spm', **kwargs):
        if seg_tool == 'spm':
            pipeline = self._spm_segmentation_pipeline(**kwargs)
        else:
            raise NotImplementedError(
                "Unrecognised segmentation tool '{}'. Can be one of 'spm'"
                .format(seg_tool))
        return pipeline

    def _spm_segmentation_pipeline(self):
        """
        Segments grey matter, white matter and CSF from T1 and T2 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self._create_pipeline(
            name='segmentation',
            inputs=['t1'],
            outputs=['t1_white_matter', 't1_grey_matter', 't1_csf',
                     't2_white_matter', 't2_grey_matter', 't2_csf'],
            description="Segment white/grey matter and csf",
            options={},
            requirements=[spm12_req],
            citations=[spm_cite],
            approx_runtime=5)
        seg = pe.Node(MultiChannelSegment(), name='seg')
        spm_path = Info.version()['path']
        tpm_path = os.path.join(spm_path, 'tpm', 'TPM.nii')
        seg.inputs.tissues = [
            ((tpm_path, 1), 5, (True, False), (False, False)),
            ((tpm_path, 2), 5, (True, False), (False, False)),
            ((tpm_path, 3), 5, (True, False), (False, False)),
            ((tpm_path, 4), 3, (False, False), (False, False)),
            ((tpm_path, 5), 4, (False, False), (False, False)),
            ((tpm_path, 6), 2, (False, False), (False, False))]
        seg.inputs.channel_info = (0, 120, (False, False),
                                   0, 120, (False, True))
        seg.inputs.affine_regularization = 'mni'
        seg.inputs.warping_regularization = [0.0, 0.001, 0.5, 0.025, 0.1]
        seg.inputs.sampling_distance = 3.0
        seg.inputs.write_deformation_fields = False
        # Not sure what inputs these should correspond to
#         seg.inputs.warping_mrf = 2.0
#         seg.inputs.warping_fwhm = 0.0
        merge = pe.Node(Merge(2), name='input_merge')
        pipeline.connect(merge, 'out', seg, 'channel_files')
        pipeline.connect_input('t2_coreg_t1', merge, 'in1')
        pipeline.connect_input('t1', merge, 'in2')
        num_channels = 2
        tissue_split = pe.Node(Split(), name='tissue_split')
        tissue_split.inputs.splits = [1] * len(seg.inputs.tissues)
        tissue_split.inputs.squeeze = True
        pipeline.connect(seg, 'native_class_images', tissue_split, 'inlist')
        channel_splits = []
        for i, tissue in enumerate(seg.inputs.tissues):
            if tissue[2][0]:
                split = pe.Node(Split(), name='tissue{}_split'.format(i))
                split.inputs.splits = [1] * num_channels
                split.inputs.squeeze = True
                pipeline.connect(tissue_split, 'out' + str(i + 1), split,
                                 'inlist')
                channel_splits.append(split)
        # Connect inputs
        pipeline.connect_input('t1', seg, 'channel_files')
        # Connect outputs
        pipeline.connect_output('t2_grey_matter', channel_splits[1], 'out1')
        pipeline.connect_output('t1_grey_matter', channel_splits[0], 'out2')
        pipeline.connect_output('t2_white_matter', channel_splits[2], 'out1')
        pipeline.connect_output('t1_white_matter', channel_splits[1], 'out2')
        pipeline.connect_output('t2_csf', channel_splits[3], 'out1')
        pipeline.connect_output('t1_csf', channel_splits[2], 'out2')
        return pipeline

    def freesurfer_pipeline(self):
        """
        Segments grey matter, white matter and CSF from T1 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self._create_pipeline(
            name='segmentation',
            inputs=['t1'],
            outputs=['freesurfer'],
            description="Segment white/grey matter and csf",
            options={},
            requirements=[freesurfer_req],
            citations=copy(freesurfer_cites),
            approx_runtime=500)
        recon_all = pe.Node(interface=ReconAll(), name='recon_all')
        recon_all.inputs.openmp = 8
        # Connect inputs/outputs
        pipeline.connect_input('t1', recon_all, 'T1_files')
        pipeline.connect_output('freesurfer', recon_all, 'subject_id')
        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format),
        DatasetSpec('freesurfer', freesurfer_format, freesurfer_pipeline),
        inherit_from=chain(MRStudy.generated_dataset_specs()))
