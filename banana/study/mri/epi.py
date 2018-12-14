from .base import MriStudy
from nipype.interfaces.fsl import TOPUP, ApplyTOPUP
from nipype.interfaces import fsl
from nipype.interfaces.utility import Merge as merge_lists
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import BET, FUGUE
from arcana.study import ParameterSpec, SwitchSpec
from arcana.data import AcquiredFilesetSpec, FilesetSpec, FieldSpec
from arcana.study.base import StudyMetaClass
from banana.citation import fsl_cite
from banana.requirement import fsl_req
from banana.interfaces.custom.motion_correction import (
    PrepareDWI, GenTopupConfigFiles)
from banana.file_format import (
    nifti_gz_format, text_matrix_format, directory_format,
    par_format, motion_mats_format, dicom_format)
from banana.interfaces.custom.fmri import FieldMapTimeInfo
from banana.interfaces.custom.motion_correction import (
    MergeListMotionMat, MotionMatCalculation)

from banana.file_format import STD_IMAGE_FORMATS


class EpiStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        AcquiredFilesetSpec('coreg_ref_wmseg', STD_IMAGE_FORMATS,
                            optional=True),
        AcquiredFilesetSpec('reverse_phase', STD_IMAGE_FORMATS, optional=True),
        AcquiredFilesetSpec('field_map_mag', STD_IMAGE_FORMATS,
                            optional=True),
        AcquiredFilesetSpec('field_map_phase', STD_IMAGE_FORMATS,
                            optional=True),
        FilesetSpec('moco', nifti_gz_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('align_mats', directory_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('moco_par', par_format,
                    'intrascan_alignment_pipeline'),
        FieldSpec('field_map_delta_te', float,
                  'field_map_time_info_pipeline')]

    add_param_specs = [
        SwitchSpec('bet_robust', True),
        ParameterSpec('bet_f_threshold', 0.2),
        ParameterSpec('bet_reduce_bias', False),
        ParameterSpec('fugue_echo_spacing', 0.000275),
        ParameterSpec('linear_coreg_method', 'epireg')]

    def linear_brain_coreg_pipeline(self, **kwargs):
        if self.branch('linear_coreg_method', 'epireg'):
            return self._epireg_linear_brain_coreg_pipeline(**kwargs)
        else:
            return super(EpiStudy, self).linear_brain_coreg_pipeline(
                **kwargs)

    def _epireg_linear_brain_coreg_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='linear_coreg',
            desc=("Intra-subjects epi registration improved using white "
                  "matter boundaries."),
            references=[fsl_cite],
            **kwargs)
        pipeline.add(
            'epireg',
            fsl.epi.EpiReg(
                out_base='epireg2ref'),
            inputs={
                'epi': ('brain', nifti_gz_format),
                't1_brain': ('coreg_ref_brain', nifti_gz_format),
                't1_head': ('coreg_ref', nifti_gz_format),
                'wmseg': ('wmseg', nifti_gz_format)},
            outputs={
                'coreg_brain': ('out_file', nifti_gz_format),
                'coreg_matrix': ('epi2str_mat', text_matrix_format)},
            requirements=[fsl_req.v('5.0.9')])

        return pipeline

    def intrascan_alignment_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='MCFLIRT_pipeline',
            desc=("Intra-epi volumes alignment."),
            references=[fsl_cite],
            **kwargs)
        mcflirt = pipeline.add(
            'mcflirt',
            fsl.MCFLIRT(
                ref_vol=0,
                save_mats=True,
                save_plots=True,
                output_type='NIFTI_GZ',
                out_file='moco.nii.gz'),
            inputs={
                'in_file': ('preproc', nifti_gz_format)},
            outputs={
                'moco': ('out_file', nifti_gz_format),
                'moco_par': ('par_file', par_format)},
            requirements=[fsl_req.v('5.0.9')])

        pipeline.add(
            'merge',
            MergeListMotionMat(),
            inputs={
                'file_list': (mcflirt, 'mat_file')},
            outputs={
                'align_mats': ('out_dir', directory_format)})

        return pipeline

    def field_map_time_info_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='field_map_time_info_pipeline',
            desc=("Pipeline to extract delta TE from field map "
                  "images, if provided"),
            version=1,
            references=[fsl_cite],
            **kwargs)

        pipeline.add(
            'extract_delta_te',
            FieldMapTimeInfo(),
            inputs={
                'fm_mag': ('field_map_mag', dicom_format)},
            outputs={
                'field_map_delta_te': ('delta_te', float)})

        return pipeline

    def preprocess_pipeline(self, **kwargs):

        if ('field_map_phase' in self.input_names and
                'field_map_mag' in self.input_names):
            return self._fugue_pipeline(**kwargs)
        elif 'reverse_phase' in self.input_names:
            return self._topup_pipeline(**kwargs)
        else:
            return super(EpiStudy, self).preprocess_pipeline(**kwargs)

    def _topup_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='preprocess_pipeline',
            desc=("Topup distortion correction pipeline"),
            references=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.add(
            'reorient_epi_in',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('magnitude', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        reorient_epi_opposite = pipeline.add(
            'reorient_epi_opposite',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('reverse_phase', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        prep_dwi = pipeline.add(
            'prepare_dwi',
            PrepareDWI(
                topup=True),
            inputs={
                'pe_dir': ('ped', str),
                'ped_polarity': ('pe_angle', str),
                'dwi': (reorient_epi_in, 'out_file'),
                'dwi1': (reorient_epi_opposite, 'out_file')})

        ped = pipeline.add(
            'gen_config',
            GenTopupConfigFiles(),
            inputs={
                'ped': (prep_dwi, 'pe')})

        merge_outputs = pipeline.add(
            'merge_files',
            merge_lists(2),
            inputs={
                'in1': (prep_dwi, 'main'),
                'in2': (prep_dwi, 'secondary')})

        merge = pipeline.add(
            'fsl_merge',
            fsl_merge(
                dimension='t'),
            inputs={
                'in_files': (merge_outputs, 'out')},
            requirements=[fsl_req.v('5.0.9')])

        topup = pipeline.add(
            'topup',
            TOPUP(),
            inputs={
                'in_file': (merge, 'merged_file'),
                'encoding_file': (ped, 'config_file')},
            requirements=[fsl_req.v('5.0.9')])

        in_apply_tp = pipeline.add(
            'in_apply_tp',
            merge_lists(1),
            inputs={
                'in1': (reorient_epi_in, 'out_file')})

        pipeline.add(
            'applytopup',
            ApplyTOPUP(
                method='jac',
                in_index=[1]),
            inputs={
                'in_files': (in_apply_tp, 'out'),
                'encoding_file': (ped, 'apply_topup_config'),
                'in_topup_movpar': (topup, 'out_movpar'),
                'in_topup_fieldcoef': (topup, 'out_fieldcoef')},
            outputs={
                'preproc': ('out_corrected', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        return pipeline

    def _fugue_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='preprocess_pipeline',
            desc=("Fugue distortion correction pipeline"),
            references=[fsl_cite],
            **kwargs)

        reorient_epi_in = pipeline.add(
            'reorient_epi_in',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('magnitude', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        fm_mag_reorient = pipeline.add(
            'reorient_fm_mag',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('field_map_mag', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        fm_phase_reorient = pipeline.add(
            'reorient_fm_phase',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('field_map_phase', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        bet = pipeline.add(
            "bet",
            BET(
                robust=True),
            inputs={
                'in_file': (fm_mag_reorient, 'out_file')},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        create_fmap = pipeline.add(
            "prepfmap",
            PrepareFieldmap(
                # delta_TE=2.46
            ),
            inputs={
                'delta_TE': ('field_map_delta_te', float),
                "in_magnitude": (bet, "out_file"),
                'in_phase': (fm_phase_reorient, 'out_file')},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        pipeline.add(
            'fugue',
            FUGUE(
                unwarp_direction='x',
                dwell_time=self.parameter('fugue_echo_spacing'),
                unwarped_file='example_func.nii.gz'),
            inputs={
                'fmap_in_file': (create_fmap, 'out_fieldmap'),
                'in_file': (reorient_epi_in, 'out_file')},
            outputs={
                'preproc': ('unwarped_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        return pipeline

    def motion_mat_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='motion_mat_calculation',
            desc=("Motion matrices calculation"),
            references=[fsl_cite],
            **kwargs)

        mm = pipeline.add(
            'motion_mats',
            MotionMatCalculation(),
            inputs={
                'reg_mat': ('coreg_matrix', text_matrix_format),
                'qform_mat': ('qform_mat', text_matrix_format)},
            outputs={
                'motion_mats': ('motion_mats', motion_mats_format)})
        if 'reverse_phase' not in self.input_names:
            pipeline.connect_input('align_mats', mm, 'align_mats',
                                   directory_format)

        return pipeline
