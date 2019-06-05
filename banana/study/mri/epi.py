from .base import MriStudy
from nipype.interfaces.fsl import TOPUP, ApplyTOPUP
from nipype.interfaces import fsl
from nipype.interfaces.utility import Merge as merge_lists
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from nipype.interfaces.fsl.epi import PrepareFieldmap
from nipype.interfaces.fsl.preprocess import BET, FUGUE
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


class EpiSeriesStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('series', STD_IMAGE_FORMATS,
                         desc=("The set of EPI volumes that make up the "
                               "series")),
        InputFilesetSpec('coreg_ref_wmseg', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('reverse_phase', STD_IMAGE_FORMATS, optional=True),
        InputFilesetSpec('field_map_mag', STD_IMAGE_FORMATS,
                         optional=True),
        InputFilesetSpec('field_map_phase', STD_IMAGE_FORMATS,
                         optional=True),
        FilesetSpec('magnitude', nifti_gz_format, 'extract_magnitude_pipeline',
                    desc=("The magnitude image, typically extracted from "
                          "the provided series")),
        FilesetSpec('series_preproc', nifti_gz_format, 'preprocess_pipeline'),
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
        SwitchSpec('bet_robust', True),
        MriStudy.param_spec('coreg_method').with_new_choices(
            'epireg', fallbacks={'epireg': 'flirt'}),
        ParamSpec('bet_f_threshold', 0.2),
        ParamSpec('bet_reduce_bias', False),
        ParamSpec('fugue_echo_spacing', 0.000275)]

    @property
    def header_image_spec_name(self):
        if self.provided('header_image'):
            hdr_name = 'header_image'
        else:
            hdr_name = 'series'
        return hdr_name

    @property
    def series_preproc_spec_name(self):
        if self.is_coregistered:
            preproc = 'series_coreg'
        else:
            preproc = 'series_preproc'
        return preproc

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
                output_type='NIFTI_GZ'),
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
            pipeline = super().coreg_brain_pipeline(**name_maps)

        return pipeline

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

    def preprocess_pipeline(self, **name_maps):

        if ('field_map_phase' in self.input_names and
                'field_map_mag' in self.input_names):
            return self._fugue_pipeline(**name_maps)
        elif 'reverse_phase' in self.input_names:
            return self._topup_pipeline(**name_maps)
        else:
            return super().preprocess_pipeline(**name_maps)

    def _topup_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='preprocess_pipeline',
            desc=("Topup distortion correction pipeline"),
            citations=[fsl_cite],
            name_maps=name_maps)

        reorient_epi_in = pipeline.add(
            'reorient_epi_in',
            fsl.utils.Reorient2Std(),
            inputs={
                'in_file': ('series', nifti_gz_format)},
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
                dimension='t',
                output_type='NIFTI_GZ'),
            inputs={
                'in_files': (merge_outputs, 'out')},
            requirements=[fsl_req.v('5.0.9')])

        topup = pipeline.add(
            'topup',
            TOPUP(
                output_type='NIFTI_GZ'),
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
                in_index=[1],
                output_type='NIFTI_GZ'),
            inputs={
                'in_files': (in_apply_tp, 'out'),
                'encoding_file': (ped, 'apply_topup_config'),
                'in_topup_movpar': (topup, 'out_movpar'),
                'in_topup_fieldcoef': (topup, 'out_fieldcoef')},
            outputs={
                'series_preproc': ('out_corrected', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.9')])

        return pipeline

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
                'in_file': ('series', nifti_gz_format)},
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
                'series_preproc': ('unwarped_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

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
