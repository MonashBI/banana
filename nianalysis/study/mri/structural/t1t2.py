import os
from nipype.pipeline import engine as pe
from itertools import chain
from nianalysis.data_formats import (
    nifti_gz_format, freesurfer_recon_all_format, text_matrix_format)
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from ...combined import CombinedStudy
from ..coregistered import CoregisteredStudy, CoregisteredToMatrixStudy
from .t1 import T1Study
from .t2 import T2Study
from nipype.interfaces.spm import Info
from nianalysis.requirements import spm12_req
from nianalysis.citations import spm_cite
from nipype.interfaces.utility import Merge, Split
from nianalysis.interfaces.spm import MultiChannelSegment


class T1T2Study(CombinedStudy):
    """
    T1 and T2 weighted MR dataset, with the T2-weighted coregistered to the T1.
    """

    sub_study_specs = {
        't1_study': (T1Study, {
            't1': 'acquired',
            'fs_recon_all': 'fs_recon_all'}),
        't2_study': (T2Study, {
            't2_coreg': 'acquired',
            'manual_wmh_mask_coreg': 'manual_wmh_mask',
            't2_masked': 'masked',
            'brain_mask': 'brain_mask'}),
        'coreg_t2_study': (CoregisteredStudy, {
            't1': 'reference',
            't2': 'to_register',
            't2_coreg': 'registered',
            't2_coreg_matrix': 'matrix'}),
        'coreg_manual_wmh_mask_study': (CoregisteredToMatrixStudy, {
            't1': 'reference',
            'manual_wmh_mask': 'to_register',
            't2_coreg_matrix': 'matrix',
            'manual_wmh_mask_coreg': 'registered'})}

    def freesurfer_pipeline(self, **kwargs):
        pipeline = self.TranslatedPipeline(
            'freesurfer', self.t1_study.freesurfer_pipeline(**kwargs), self,
            add_inputs=['t2_coreg'])
        recon_all = pipeline.node('recon_all')
        recon_all.inputs.use_T2 = True
        # Connect T2-weighted input
        pipeline.connect_input('t2_coreg', recon_all, 'T2_file')
        pipeline.assert_connected()
        return pipeline

    t2_registration_pipeline = CombinedStudy.translate(
        'coreg_t2_study', CoregisteredStudy.registration_pipeline)

    manual_wmh_mask_registration_pipeline = CombinedStudy.translate(
        'coreg_t2_study', CoregisteredToMatrixStudy.registration_pipeline)

    t2_brain_mask_pipeline = CombinedStudy.translate(
        'coreg_t2_study', T2Study.brain_mask_pipeline)

    def segmentation_pipeline(self, seg_tool='spm', **kwargs):
        if seg_tool == 'spm':
            pipeline = self._spm_segmentation_pipeline(**kwargs)
        else:
            raise NotImplementedError(
                "Unrecognised segmentation tool '{}'. Can be one of 'spm'"
                .format(seg_tool))
        return pipeline

    def _spm_segmentation_pipeline(self, **kwargs):  # @UnusedVariable
        """
        Segments grey matter, white matter and CSF from T1 and T2 images using
        SPM "NewSegment" function.

        NB: Default values come from the W2MHS toolbox
        """
        pipeline = self._create_pipeline(
            name='segmentation',
            inputs=['acquired'],
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

    _dataset_specs = set_dataset_specs(
        DatasetSpec('t1', nifti_gz_format,
                    description="Raw T1-weighted image (e.g. MPRAGE)"),
        DatasetSpec('t2', nifti_gz_format,
                    description="Raw T2-weighted image (e.g. FLAIR)"),
        DatasetSpec('manual_wmh_mask', nifti_gz_format,
                    description="Manual WMH segmentations"),
        DatasetSpec('t2_coreg', nifti_gz_format, t2_registration_pipeline,
                    description="T2 registered to T1 weighted"),
        DatasetSpec('t1_masked', nifti_gz_format, t1_brain_mask_pipeline,
                    description="T1 masked by brain mask"),
        DatasetSpec('t2_masked', nifti_gz_format, t2_brain_mask_pipeline,
                    description="Coregistered T2 masked by brain mask"),
        DatasetSpec('brain_mask', nifti_gz_format, t2_brain_mask_pipeline,
                    description="Brain mask generated from coregistered T2"),
        DatasetSpec('manual_wmh_mask_coreg', nifti_gz_format,
                    manual_wmh_mask_registration_pipeline,
                    description="Manual WMH segmentations coregistered to T1"),
        DatasetSpec('t2_coreg_matrix', text_matrix_format,
                    t2_registration_pipeline,
                    description="Coregistration matrix for T2 to T1"),
        DatasetSpec('fs_recon_all', freesurfer_recon_all_format,
                    freesurfer_pipeline,
                    description="Output directory from Freesurfer recon_all"))
