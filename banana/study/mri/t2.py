from arcana.study.base import StudyMetaClass
from .base import MriStudy
from arcana.parameter import ParameterSpec
from copy import copy
from nipype.interfaces.freesurfer.preprocess import ReconAll
# from arcana.interfaces.utils import DummyReconAll as ReconAll
from banana.requirement import freesurfer_req, ants19_req, fsl5_req
from banana.citation import freesurfer_cites, fsl_cite
from nipype.interfaces import fsl, ants
from arcana.interfaces import utils
from banana.file_format import (
    freesurfer_recon_all_format, nifti_gz_format, text_matrix_format)
from arcana.data import FilesetSpec
from arcana.interfaces.utils import JoinPath
from .base import MriStudy
from arcana.study.base import StudyMetaClass
from arcana.parameter import ParameterSpec


class T2Study(MriStudy, metaclass=StudyMetaClass):

    add_param_specs = [
        ParameterSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.5),
        ParameterSpec('bet_reduce_bias', False)]

    def cet_T2s(self, **options):
        pipeline = self.pipeline(
            name='CET_T2s',
            inputs=[FilesetSpec('betted_T2s', nifti_gz_format),
                    FilesetSpec('betted_T2s_mask', nifti_gz_format),
                    FilesetSpec('betted_T2s_last_echo', nifti_gz_format),
                    FilesetSpec(
                self._lookup_nl_tfm_inv_name('SUIT'),
                nifti_gz_format),
                FilesetSpec(
                self._lookup_l_tfm_to_name('SUIT'),
                text_matrix_format),
                FilesetSpec('T2s_to_T1_mat', text_matrix_format)],
            outputs=[FilesetSpec('cetted_T2s_mask', nifti_gz_format),
                     FilesetSpec('cetted_T2s', nifti_gz_format),
                     FilesetSpec('cetted_T2s_last_echo', nifti_gz_format)],
            desc=("Construct cerebellum mask using SUIT template"),
            default_options={
                'SUIT_mask': self._lookup_template_mask_path('SUIT')},
            citations=[fsl_cite],
            options=options)

        # Initially use MNI space to warp SUIT mask into T2s space
        merge_trans = pipeline.create_node(
            utils.Merge(3), name='merge_transforms')
        pipeline.connect_input(
            self._lookup_nl_tfm_inv_name('SUIT'),
            merge_trans,
            'in3')
        pipeline.connect_input(
            self._lookup_l_tfm_to_name('SUIT'),
            merge_trans,
            'in2')
        pipeline.connect_input('T2s_to_T1_mat', merge_trans, 'in1')

        apply_trans = pipeline.create_node(
            ants.resampling.ApplyTransforms(), name='ApplyTransform',
            requirements=[ants19_req], memory=16000, wall_time=120)
        apply_trans.inputs.interpolation = 'NearestNeighbor'
        apply_trans.inputs.input_image_type = 3
        apply_trans.inputs.invert_transform_flags = [True, True, False]
        apply_trans.inputs.input_image = pipeline.option('SUIT_mask')

        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('betted_T2s', apply_trans, 'reference_image')

        # Combine masks
        maths1 = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_optiBET_masks', op_string='-mas'),
            name='combine_masks', requirements=[fsl5_req], memory=16000,
            wall_time=5)
        pipeline.connect_input('betted_T2s_mask', maths1, 'in_file')
        pipeline.connect(apply_trans, 'output_image', maths1, 'in_file2')

        # Mask out t2s image
        maths2 = pipeline.create_node(
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas'),
            name='mask_t2s', requirements=[fsl5_req], memory=16000,
            wall_time=5)
        pipeline.connect_input('betted_T2s', maths2, 'in_file')
        pipeline.connect(maths1, 'output_image', maths2, 'in_file2')

        maths3 = pipeline.create_node(
            fsl.utils.ImageMaths(
                suffix='_optiBET_cerebellum',
                op_string='-mas'),
            name='mask_t2s_last_echo', requirements=[fsl5_req],
            memory=16000, wall_time=5)
        pipeline.connect_input('betted_T2s_last_echo', maths3, 'in_file')
        pipeline.connect(maths1, 'output_image', maths3, 'in_file2')

        pipeline.connect_output('cetted_T2s', maths2, 'out_file')
        pipeline.connect_output('cetted_T2s_mask', apply_trans,
                                'output_image')
        pipeline.connect_output('cetted_T2s_last_echo', maths3,
                                'out_file')

        return pipeline

    def bet_T2s(self, **options):

        pipeline = self.pipeline(
            name='BET_T2s',
            inputs=[FilesetSpec('t2s', nifti_gz_format),
                    FilesetSpec('t2s_last_echo', nifti_gz_format)],
            outputs=[FilesetSpec('betted_T2s', nifti_gz_format),
                     FilesetSpec('betted_T2s_mask', nifti_gz_format),
                     FilesetSpec('betted_T2s_last_echo', nifti_gz_format)],
            desc=("python implementation of BET"),
            default_options={},
            citations=[fsl_cite],
            options=options)

        bet = pipeline.create_node(
            fsl.BET(frac=0.1, mask=True), name='bet',
            requirements=[fsl5_req], memory=8000, wall_time=45)
        pipeline.connect_input('t2s', bet, 'in_file')
        pipeline.connect_output('betted_T2s', bet, 'out_file')
        pipeline.connect_output('betted_T2s_mask', bet, 'mask_file')

        maths = pipeline.create_node(
            fsl.utils.ImageMaths(suffix='_BET_brain', op_string='-mas'),
            name='mask', requirements=[fsl5_req], memory=16000, wall_time=5)
        pipeline.connect_input('t2s_last_echo', maths, 'in_file')
        pipeline.connect(bet, 'mask_file', maths, 'in_file2')
        pipeline.connect_output('betted_T2s_last_echo', maths, 'out_file')

        return pipeline
