from .base import MriStudy
from nipype.interfaces.fsl import (
    TOPUP, ApplyTOPUP, Merge as FslMerge, BET, FUGUE, PrepareFieldmap)
from nipype.interfaces import fsl
from nipype.interfaces.utility import Merge as merge_lists
from arcana.study import ParamSpec, SwitchSpec
from arcana.data import InputFilesetSpec, FilesetSpec, FieldSpec
from arcana.study.base import StudyMetaClass
from banana.citation import fsl_cite
from banana.requirement import fsl_req
from banana.interfaces.custom.motion_correction import (
    PrepareDWI, GenTopupConfigFiles)
from banana.file_format import (
    nifti_gz_format, text_matrix_format,
    par_format, motion_mats_format, dicom_format)
from banana.interfaces.custom.bold import FieldMapTimeInfo
from banana.interfaces.custom.motion_correction import (
    MergeListMotionMat, MotionMatCalculation)
from banana.exceptions import BananaUsageError
from banana.file_format import STD_IMAGE_FORMATS
from banana.interfaces.mrtrix import MRConvert
from banana.requirement import mrtrix_req


class EpiStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('coreg_ref_wmseg', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('field_map_mag', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('field_map_phase', STD_IMAGE_FORMATS,
                         optional=True),
        FieldSpec('field_map_delta_te', float,
                  'field_map_time_info_pipeline')]

    add_param_specs = [
        SwitchSpec('bet_robust', True),
        ParamSpec('bet_f_threshold', 0.2),
        ParamSpec('bet_reduce_bias', False),
        ParamSpec('fugue_echo_spacing', 0.000275)]

    def preprocess_pipeline(self, **name_maps):

        if ('field_map_phase' in self.input_names and
                'field_map_mag' in self.input_names):
            return self._fugue_pipeline(**name_maps)
        else:
            return super().preprocess_pipeline(**name_maps)

    def _fugue_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='preprocess_pipeline',
            desc=("Fugue distortion correction pipeline"),
            citations=[fsl_cite],
            name_maps=name_maps)

        reorient_epi_in = pipeline.add(
            'reorient_epi_in',
            fsl.utils.Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('magnitude', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        fm_mag_reorient = pipeline.add(
            'reorient_fm_mag',
            fsl.utils.Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('field_map_mag', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        fm_phase_reorient = pipeline.add(
            'reorient_fm_phase',
            fsl.utils.Reorient2Std(
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('field_map_phase', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        bet = pipeline.add(
            "bet",
            BET(
                robust=True,
                output_type='NIFTI_GZ'),
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
                unwarped_file='example_func.nii.gz',
                output_type='NIFTI_GZ'),
            inputs={
                'fmap_in_file': (create_fmap, 'out_fieldmap'),
                'in_file': (reorient_epi_in, 'out_file')},
            outputs={
                'mag_preproc': ('unwarped_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        return pipeline

    def field_map_time_info_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='field_map_time_info_pipeline',
            desc=("Pipeline to extract delta TE from field map "
                  "images, if provided"),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            'extract_delta_te',
            FieldMapTimeInfo(),
            inputs={
                'fm_mag': ('field_map_mag', dicom_format)},
            outputs={
                'field_map_delta_te': ('delta_te', float)})

        return pipeline


class EpiSeriesStudy(EpiStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('series', STD_IMAGE_FORMATS,
                         desc=("The set of EPI volumes that make up the "
                               "series")),
        InputFilesetSpec('coreg_ref_wmseg', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('field_map_mag', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('field_map_phase', STD_IMAGE_FORMATS,
                         optional=True),
        FilesetSpec('magnitude', nifti_gz_format, 'extract_magnitude_pipeline',
                    desc=("The magnitude image, typically extracted from "
                          "the provided series")),
        FilesetSpec('series_preproc', nifti_gz_format, 'preprocess_pipeline'),
        FilesetSpec('mag_preproc', nifti_gz_format, 'mag_preproc_pipeline'),
        FilesetSpec('series_coreg', nifti_gz_format, 'series_coreg_pipeline'),
        FilesetSpec('moco', nifti_gz_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('align_mats', motion_mats_format,
                    'intrascan_alignment_pipeline'),
        FilesetSpec('moco_par', par_format,
                    'intrascan_alignment_pipeline'),
        FieldSpec('field_map_delta_te', float,
                  'field_map_time_info_pipeline')]

    add_param_specs = [
        MriStudy.param_spec('coreg_method').with_new_choices(
            'epireg', fallbacks={'epireg': 'flirt'})]

    primary_scan_name = 'series'

    @property
    def series_preproc_spec_name(self):
        if self.is_coregistered:
            preproc = 'series_coreg'
        else:
            preproc = 'series_preproc'
        return preproc

    def extract_magnitude_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            'extract_magnitude',
            desc="Extracts a single magnitude volume from a series",
            citations=[],
            name_maps=name_maps)

        pipeline.add(
            "extract_first_vol",
            MRConvert(
                coord=(3, 0)),
            inputs={
                'in_file': ('series', nifti_gz_format)},
            outputs={
                'magnitude': ('out_file', nifti_gz_format)},
            requirements=[mrtrix_req.v('3.0rc3')])

        return pipeline

    def preprocess_pipeline(self, **name_maps):
        return super().preprocess_pipeline(
            input_map={'magnitude': 'series'},
            output_map={'mag_preproc': 'series_preproc'},
            name_maps=name_maps)

    def mag_preproc_pipeline(self, **name_maps):
        return self.extract_magnitude_pipeline(
            input_map={'series': 'series_preproc'},
            output_map={'magnitude': 'mag_preproc'},
            name_maps=name_maps)

    def coreg_pipeline(self, **name_maps):
        if self.branch('coreg_method', 'epireg'):
            pipeline = self._epireg_linear_coreg_pipeline(**name_maps)
        else:
            pipeline = super().coreg_pipeline(**name_maps)
        return pipeline

    def _epireg_linear_coreg_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='linear_coreg',
            desc=("Intra-subjects epi registration improved using white "
                  "matter boundaries."),
            citations=[fsl_cite],
            name_maps=name_maps)

        epireg = pipeline.add(
            'epireg',
            fsl.epi.EpiReg(
                out_base='epireg2ref',
                output_type='NIFTI_GZ',
                no_clean=True),
            inputs={
                'epi': ('brain', nifti_gz_format),
                't1_brain': ('coreg_ref_brain', nifti_gz_format),
                't1_head': ('coreg_ref', nifti_gz_format)},
            outputs={
                'brain_coreg': ('out_file', nifti_gz_format),
                'coreg_fsl_mat': ('epi2str_mat', text_matrix_format)},
            requirements=[fsl_req.v('5.0.9')])

        if self.provided('coreg_ref_wmseg'):
            pipeline.connect_input('coreg_ref_wmseg', epireg, 'wmseg',
                                   nifti_gz_format)

        return pipeline

    def brain_coreg_pipeline(self, **name_maps):
        if self.branch('coreg_method', 'epireg'):
            pipeline = self.coreg_pipeline(
                name='brain_coreg',
                name_maps=dict(
                    input_map={
                        'mag_preproc': 'brain',
                        'coreg_ref': 'coreg_ref_brain'},
                    output_map={
                        'mag_coreg': 'brain_coreg'},
                    name_maps=name_maps))

            pipeline.add(
                'mask_transform',
                fsl.ApplyXFM(
                    output_type='NIFTI_GZ',
                    apply_xfm=True),
                inputs={
                    'in_matrix_file': (pipeline.node('epireg'), 'epi2str_mat'),
                    'in_file': ('brain_mask', nifti_gz_format),
                    'reference': ('coreg_ref_brain', nifti_gz_format)},
                outputs={
                    'brain_mask_coreg': ('out_file', nifti_gz_format)},
                requirements=[fsl_req.v('5.0.10')],
                wall_time=10)
        else:
            pipeline = super().brain_coreg_pipeline(**name_maps)

        return pipeline

    def series_coreg_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            'series_coreg',
            desc="Applies coregistration transform to DW series",
            citations=[],
            name_maps=name_maps)

        if self.provided('coreg_ref'):
            coreg_ref = 'coreg_ref'
        elif self.provided('coreg_ref_brain'):
            coreg_ref = 'coreg_ref_brain'
        else:
            raise BananaUsageError(
                "Cannot coregister DW series as reference ('coreg_ref' or "
                "'coreg_ref_brain') has not been provided to {}".format(self))

        # Apply co-registration transformation to DW series
        pipeline.add(
            'mask_transform',
            fsl.ApplyXFM(
                output_type='NIFTI_GZ',
                apply_xfm=True),
            inputs={
                'in_matrix_file': ('coreg_fsl_mat', text_matrix_format),
                'in_file': ('series_preproc', nifti_gz_format),
                'reference': (coreg_ref, nifti_gz_format)},
            outputs={
                'series_coreg': ('out_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=10)

        return pipeline

    def intrascan_alignment_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='MCFLIRT_pipeline',
            desc=("Intra-epi volumes alignment."),
            citations=[fsl_cite],
            name_maps=name_maps)

        mcflirt = pipeline.add(
            'mcflirt',
            fsl.MCFLIRT(
                ref_vol=0,
                save_mats=True,
                save_plots=True,
                output_type='NIFTI_GZ',
                out_file='moco.nii.gz'),
            inputs={
                'in_file': ('mag_preproc', nifti_gz_format)},
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
                'align_mats': ('out_dir', motion_mats_format)})

        return pipeline

    def motion_mat_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='motion_mat_calculation',
            desc=("Motion matrices calculation"),
            citations=[fsl_cite],
            name_maps=name_maps)

        mm = pipeline.add(
            'motion_mats',
            MotionMatCalculation(),
            inputs={
                'reg_mat': ('coreg_fsl_mat', text_matrix_format),
                'qform_mat': ('qform_mat', text_matrix_format)},
            outputs={
                'motion_mats': ('motion_mats', motion_mats_format)})
        if 'reverse_phase' not in self.input_names:
            pipeline.connect_input('align_mats', mm, 'align_mats',
                                   motion_mats_format)

        return pipeline
