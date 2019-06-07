from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.afni.preprocess import Volreg
from nipype.interfaces.fsl.utils import ImageMaths, ConvertXFM
from banana.interfaces.fsl import (FSLFIX, FSLFixTraining,
                                       SignalRegression, PrepareFIXTraining)
from arcana.data import FilesetSpec, InputFilesetSpec
from arcana.study.base import StudyMetaClass
from banana.requirement import (
    afni_req, fix_req, fsl_req, ants_req, c3d_req)
from banana.citation import fsl_cite
from banana.file_format import (
    nifti_gz_format, nifti_gz_x_format, rfile_format, directory_format,
    zip_format, par_format, text_format, dicom_format, text_matrix_format)
from banana.interfaces.afni import Tproject
from nipype.interfaces.utility import Merge as NiPypeMerge
import os.path as op
from nipype.interfaces.utility.base import IdentityInterface
from arcana.study import ParamSpec
from nipype.interfaces.ants.resampling import ApplyTransforms
from banana.study.mri.t1 import T1Study
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from arcana.data import InputFilesets
from arcana.utils.interfaces import CopyToDir
from nipype.interfaces.afni.preprocess import BlurToFWHM
from banana.interfaces.custom.bold import PrepareFIX
from banana.interfaces.c3d import ANTs2FSLMatrixConversion
import logging
from arcana.exceptions import ArcanaNameError
from banana.bids_ import BidsInput, BidsAssocInput
from .epi import EpiSeriesStudy

logger = logging.getLogger('banana')


atlas_path = op.abspath(
    op.join(op.dirname(__file__), '..', '..', '..', 'atlases'))

IMAGE_TYPE_TAG = ('0008', '0008')
PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']
MAG_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']


class BoldStudy(EpiSeriesStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        InputFilesetSpec('train_data', rfile_format, optional=True,
                         frequency='per_study'),
        FilesetSpec('hand_label_noise', text_format,
                    'fix_preparation_pipeline'),
        FilesetSpec('labelled_components', text_format,
                    'fix_classification_pipeline'),
        FilesetSpec('cleaned_file', nifti_gz_format,
                    'fix_regression_pipeline'),
        FilesetSpec('filtered_data', nifti_gz_format,
                    'rsfMRI_filtering_pipeline'),
        FilesetSpec('mc_par', par_format, 'rsfMRI_filtering_pipeline'),
        FilesetSpec('melodic_ica', zip_format,
                    'single_subject_melodic_pipeline'),
        FilesetSpec('fix_dir', zip_format, 'fix_preparation_pipeline'),
        FilesetSpec('normalized_ts', nifti_gz_format,
                    'timeseries_normalization_to_atlas_pipeline'),
        FilesetSpec('smoothed_ts', nifti_gz_format,
                    'smoothing_pipeline')]

    add_param_specs = [
        ParamSpec('component_threshold', 20),
        ParamSpec('motion_reg', True),
        ParamSpec('highpass', 0.01),
        ParamSpec('brain_thresh_percent', 5),
        ParamSpec('group_ica_components', 15)]

    primary_bids_selector = BidsInput(
        spec_name='series', type='bold', format=nifti_gz_x_format)

    default_bids_inputs = [primary_bids_selector,
                           BidsAssocInput(
                               spec_name='field_map_phase',
                               primary=primary_bids_selector,
                               association='phasediff',
                               format=nifti_gz_format,
                               drop_if_missing=True),
                           BidsAssocInput(
                               spec_name='field_map_mag',
                               primary=primary_bids_selector,
                               association='phasediff',
                               type='magnitude',
                               format=nifti_gz_format,
                               drop_if_missing=True)]

    def rsfMRI_filtering_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='rsfMRI_filtering',
            desc=("Spatial and temporal rsfMRI filtering"),
            citations=[fsl_cite],
            name_maps=name_maps)

        afni_mc = pipeline.add(
            'AFNI_MC',
            Volreg(
                zpad=1,
                out_file='rsfmri_mc.nii.gz',
                oned_file='prefiltered_func_data_mcf.par'),
            inputs={
                'in_file': ('series_preproc', nifti_gz_format)},
            outputs={
                'mc_par': ('oned_file', par_format)},
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])

        filt = pipeline.add(
            'Tproject',
            Tproject(
                stopband=(0, 0.01),
                polort=3,
                blur=3,
                out_file='filtered_func_data.nii.gz'),
            inputs={
                'delta_t': ('tr', float),
                'mask': (self.brain_mask_spec_name, nifti_gz_format),
                'in_file': (afni_mc, 'out_file')},
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])

        meanfunc = pipeline.add(
            'meanfunc',
            ImageMaths(
                op_string='-Tmean',
                suffix='_mean',
                output_type='NIFTI_GZ'),
            wall_time=5,
            inputs={
                'in_file': (afni_mc, 'out_file')},
            requirements=[fsl_req.v('5.0.10')])

        pipeline.add(
            'add_mean',
            ImageMaths(
                op_string='-add',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (filt, 'out_file'),
                'in_file2': (meanfunc, 'out_file')},
            outputs={
                'filtered_data': ('out_file', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.10')])

        return pipeline

    def single_subject_melodic_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='MelodicL1',
            desc=("Single subject ICA analysis using FSL MELODIC."),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            'melodic_L1',
            MELODIC(
                no_bet=True,
                bg_threshold=self.parameter('brain_thresh_percent'),
                report=True,
                out_stats=True,
                mm_thresh=0.5,
                out_dir='melodic_ica',
                output_type='NIFTI_GZ'),
            inputs={
                'mask': (self.brain_mask_spec_name, nifti_gz_format),
                'tr_sec': ('tr', float),
                'in_files': ('filtered_data', nifti_gz_format)},
            outputs={
                'melodic_ica': ('out_dir', directory_format)},
            wall_time=15,
            requirements=[fsl_req.v('5.0.10')])

        return pipeline

    def fix_preparation_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='prepare_fix',
            desc=("Pipeline to create the right folder structure before "
                  "running FIX"),
            citations=[fsl_cite],
            name_maps=name_maps)

        if self.branch('coreg_to_tmpl_method', 'ants'):

            struct_ants2fsl = pipeline.add(
                'struct_ants2fsl',
                ANTs2FSLMatrixConversion(
                    ras2fsl=True),
                inputs={
                    'reference_file': ('template_brain', nifti_gz_format),
                    'itk_file': ('coreg_to_tmpl_ants_mat', text_matrix_format),
                    'source_file': ('coreg_ref_brain', nifti_gz_format)},
                requirements=[c3d_req.v('1.0.0')])

            struct_matrix = (struct_ants2fsl, 'fsl_matrix')
        else:
            struct_matrix = ('coreg_to_tmpl_fsl_mat', text_matrix_format)

#         if self.branch('coreg_method', 'ants'):
#         epi_ants2fsl = pipeline.add(
#             'epi_ants2fsl',
#             ANTs2FSLMatrixConversion(
#                 ras2fsl=True),
#             inputs={
#                 'source_file': ('brain', nifti_gz_format),
#                 'itk_file': ('coreg_ants_mat', text_matrix_format),
#                 'reference_file': ('coreg_ref_brain', nifti_gz_format)},
#             requirements=[c3d_req.v('1.0.0')])

        MNI2t1 = pipeline.add(
            'MNI2t1',
            ConvertXFM(
                invert_xfm=True),
            inputs={
                'in_file': struct_matrix},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        struct2epi = pipeline.add(
            'struct2epi',
            ConvertXFM(
                invert_xfm=True),
            inputs={
                'in_file': ('coreg_fsl_mat', text_matrix_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        meanfunc = pipeline.add(
            'meanfunc',
            ImageMaths(
                op_string='-Tmean',
                suffix='_mean',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('series_preproc', nifti_gz_format)},
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])

        pipeline.add(
            'prep_fix',
            PrepareFIX(),
            inputs={
                'melodic_dir': ('melodic_ica', directory_format),
                't1_brain': ('coreg_ref_brain', nifti_gz_format),
                'mc_par': ('mc_par', par_format),
                'epi_brain_mask': ('brain_mask', nifti_gz_format),
                'epi_preproc': ('series_preproc', nifti_gz_format),
                'filtered_epi': ('filtered_data', nifti_gz_format),
                'epi2t1_mat': ('coreg_fsl_mat', text_matrix_format),
                't12MNI_mat': (struct_ants2fsl, 'fsl_matrix'),
                'MNI2t1_mat': (MNI2t1, 'out_file'),
                't12epi_mat': (struct2epi, 'out_file'),
                'epi_mean': (meanfunc, 'out_file')},
            outputs={
                'fix_dir': ('fix_dir', directory_format),
                'hand_label_noise': ('hand_label_file', text_format)})

        return pipeline

    def fix_classification_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='fix_classification',
            desc=("Automatic classification of noisy components from the "
                  "rsfMRI data using fsl FIX."),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            "fix",
            FSLFIX(
                component_threshold=self.parameter('component_threshold'),
                motion_reg=self.parameter('motion_reg'),
                classification=True),
            inputs={
                "feat_dir": ("fix_dir", directory_format),
                "train_data": ("train_data", rfile_format)},
            outputs={
                'labelled_components': ('label_file', text_format)},
            wall_time=30,
            requirements=[fsl_req.v('5.0.9'), fix_req.v('1.0')])

        return pipeline

    def fix_regression_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='signal_regression',
            desc=("Regression of the noisy components from the rsfMRI data "
                  "using a python implementation equivalent to that in FIX."),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            "signal_reg",
            SignalRegression(
                motion_regression=self.parameter('motion_reg'),
                highpass=self.parameter('highpass')),
            inputs={
                "fix_dir": ("fix_dir", directory_format),
                "labelled_components": ("labelled_components", text_format)},
            outputs={
                'cleaned_file': ('output', nifti_gz_format)},
            wall_time=30,
            requirements=[fsl_req.v('5.0.9'), fix_req.v('1.0')])

        return pipeline

    def timeseries_normalization_to_atlas_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='timeseries_normalization_to_atlas_pipeline',
            desc=("Apply ANTs transformation to the fmri filtered file to "
                  "normalize it to MNI 2mm."),
            citations=[fsl_cite],
            name_maps=name_maps)

        merge_trans = pipeline.add(
            'merge_transforms',
            NiPypeMerge(3),
            inputs={
                'in1': ('coreg_to_tmpl_ants_warp', nifti_gz_format),
                'in2': ('coreg_to_tmpl_ants_mat', text_matrix_format),
                'in3': ('coreg_matrix', text_matrix_format)},
            wall_time=1)

        pipeline.add(
            'ApplyTransform',
            ApplyTransforms(
                interpolation='Linear',
                input_image_type=3),
            inputs={
                'reference_image': ('template_brain', nifti_gz_format),
                'input_image': ('cleaned_file', nifti_gz_format),
                'transforms': (merge_trans, 'out')},
            outputs={
                'normalized_ts': ('output_image', nifti_gz_format)},
            wall_time=7,
            mem_gb=24,
            requirements=[ants_req.v('2')])

        return pipeline

    def smoothing_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='smoothing_pipeline',
            desc=("Spatial smoothing of the normalized fmri file"),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            '3dBlurToFWHM',
            BlurToFWHM(
                fwhm=5,
                out_file='smoothed_ts.nii.gz'),
            inputs={
                'mask': ('template_mask', nifti_gz_format),
                'in_file': ('normalized_ts', nifti_gz_format)},
            outputs={
                'smoothed_ts': ('out_file', nifti_gz_format)},
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])

        return pipeline


class MultiBoldMixin(MultiStudy):
    """
    A mixin class used for studies with an array of fMRI scans. Can be used to
    perform combined analysis over the array
    """

    add_data_specs = [
        FilesetSpec('train_data', rfile_format, 'fix_training_pipeline',
                    frequency='per_study'),
        FilesetSpec('fmri_pre-processeing_results', directory_format,
                    'gather_fmri_result_pipeline'),
        FilesetSpec('group_melodic', directory_format,
                    'group_melodic_pipeline')]

    @classmethod
    def fmri_substudies(cls):
        # Detect fMRI sub-studies
        fmri_substudies = []
        for substudy_spec in cls.substudy_specs():
            try:
                cls.data_spec(substudy_spec.inverse_map('fix_dir'))
            except ArcanaNameError:
                continue  # Sub study is not a Fmri study
            else:
                fmri_substudies.append(substudy_spec.name)

    def fix_training_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='training_fix',
            desc=("Pipeline to create the training set for FIX given a group "
                  "of subjects with the hand_label_noise.txt file within "
                  "their fix_dir."),
            citations=[fsl_cite],
            name_maps=name_maps)

        num_fix_dirs = len(self.bold_substudies())
        merge_fix_dirs = pipeline.add(
            'merge_fix_dirs',
            NiPypeMerge(num_fix_dirs))
        merge_label_files = pipeline.add(
            'merge_label_files',
            NiPypeMerge(num_fix_dirs))
        for i, substudy_name in enumerate(self.bold_substudies(), start=1):
            spec = self.substudy_spec(substudy_name)
            pipeline.connect_input(
                spec.inverse_map('fix_dir'), merge_fix_dirs, 'in{}'.format(i),
                directory_format)
            pipeline.connect_input(
                spec.inverse_map('hand_label_noise'), merge_label_files,
                'in{}'.format(i), text_format)

        merge_visits = pipeline.add(
            IdentityInterface(
                ['list_dir', 'list_label_files']),
            inputs={
                'list_dir': (merge_fix_dirs, 'out'),
                'list_label_files': (merge_label_files, 'out')},
            joinsource=self.SUBJECT_ID,
            joinfield=['list_dir', 'list_label_files'], name='merge_visits')

        merge_subjects = pipeline.add(
            'merge_subjects',
            NiPypeMerge(
                2,
                ravel_inputs=True),
            inputs={
                'in1': (merge_visits, 'list_dir'),
                'in2': (merge_visits, 'list_label_files')},
            joinsource=self.SUBJECT_ID,
            joinfield=['in1', 'in2'])

        prepare_training = pipeline.add(
            'prepare_training',
            PrepareFIXTraining(
                epi_number=num_fix_dirs),
            inputs={
                'inputs_list': (merge_subjects, 'out')})

        pipeline.add(
            'fix_training',
            FSLFixTraining(
                outname='FIX_training_set',
                training=True),
            inputs={
                'list_dir': (prepare_training, 'prepared_dirs')},
            outputs={
                'train_data': ('training_set', rfile_format)},
            wall_time=240,
            requirements=[fix_req.v('1.0')])

        return pipeline

    def gather_fmri_result_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='gather_fmri',
            desc=("Pipeline to gather together all the pre-processed "
                  "fMRI images"),
            version=1,
            citations=[fsl_cite],
            name_maps=name_maps)

        merge_inputs = pipeline.add(
            'merge_inputs',
            NiPypeMerge(len(self.bold_substudies())))
        for i, substudy_name in enumerate(self.bold_substudies(), start=1):
            spec = self.substudy_spec(substudy_name)
            pipeline.connect_input(
                spec.inverse_map('smoothed_ts'),
                merge_inputs, 'in{}'.format(i), nifti_gz_format)

        pipeline.add(
            'copy2dir',
            CopyToDir(),
            inputs={
                'in_files': (merge_inputs, 'out')},
            outputs={
                'fmri_pre-processeing_results': ('out_dir', directory_format)})

        return pipeline

    def group_melodic_pipeline(self, **name_maps):

        pipeline = self.new_pipeline(
            name='group_melodic',
            desc=("Group ICA"),
            citations=[fsl_cite],
            name_maps=name_maps)

        pipeline.add(
            MELODIC(
                no_bet=True,
                bg_threshold=self.parameter('brain_thresh_percent'),
                dim=self.parameter('group_ica_components'),
                report=True,
                out_stats=True,
                mm_thresh=0.5,
                sep_vn=True,
                out_dir='group_melodic.ica',
                output_type='NIFTI_GZ'),
            inputs={
                'bg_image': ('template_brain', nifti_gz_format),
                'mask': ('template_mask', nifti_gz_format),
                'in_files': ('smoothed_ts', nifti_gz_format),
                'tr_sec': ('tr', float)},
            outputs={
                'group_melodic': ('out_dir', directory_format)},
            joinsource=self.SUBJECT_ID,
            joinfield=['in_files'],
            name='gica',
            requirements=[fsl_req.v('5.0.10')],
            wall_time=7200)

        return pipeline


def create_multi_fmri_class(name, t1, epis, epi_number, echo_spacing,
                            fm_mag=None, fm_phase=None, run_regression=False):

    inputs = []
    dct = {}
    data_specs = []
    param_specs = []
    output_files = []
    distortion_correction = False

    if fm_mag and fm_phase:
        logger.info(
            'Both magnitude and phase field map images provided. EPI '
            'ditortion correction will be performed.')
        distortion_correction = True
    elif fm_mag or fm_phase:
        logger.info(
            'In order to perform EPI ditortion correction both magnitude '
            'and phase field map images must be provided.')
    else:
        logger.info(
            'No field map image provided. Distortion correction will not be'
            'performed.')

    study_specs = [SubStudySpec('t1', T1Study)]
    ref_spec = {'t1_brain': 'coreg_ref_brain'}
    inputs.append(InputFilesets('t1_primary', t1, dicom_format,
                                  is_regex=True, order=0))
    epi_refspec = ref_spec.copy()
    epi_refspec.update({'t1_wm_seg': 'coreg_ref_wmseg',
                        't1_preproc': 'coreg_ref',
                        'train_data': 'train_data'})
    study_specs.append(SubStudySpec('epi_0', BoldStudy, epi_refspec))
    if epi_number > 1:
        epi_refspec.update({'t1_wm_seg': 'coreg_ref_wmseg',
                            't1_preproc': 'coreg_ref',
                            'train_data': 'train_data',
                            'epi_0_coreg_to_tmpl_warp': 'coreg_to_tmpl_warp',
                            'epi_0_coreg_to_tmpl_ants_mat':
                            'coreg_to_tmpl_ants_mat'})
        study_specs.extend(SubStudySpec('epi_{}'.format(i), BoldStudy,
                                        epi_refspec)
                           for i in range(1, epi_number))

    study_specs.extend(SubStudySpec('epi_{}'.format(i), BoldStudy,
                                    epi_refspec)
                       for i in range(epi_number))

    for i in range(epi_number):
        inputs.append(InputFilesets(
            'epi_{}_primary'.format(i), epis, dicom_format, order=i,
            is_regex=True))
#     inputs.extend(InputFilesets(
#         'epi_{}_hand_label_noise'.format(i), text_format,
#         'hand_label_noise_{}'.format(i+1))
#         for i in range(epi_number))
        param_specs.append(
            ParamSpec('epi_{}_fugue_echo_spacing'.format(i), echo_spacing))

    if distortion_correction:
        inputs.extend(InputFilesets(
            'epi_{}_field_map_mag'.format(i), fm_mag, dicom_format,
            dicom_tags={IMAGE_TYPE_TAG: MAG_IMAGE_TYPE}, is_regex=True,
            order=0)
            for i in range(epi_number))
        inputs.extend(InputFilesets(
            'epi_{}_field_map_phase'.format(i), fm_phase, dicom_format,
            dicom_tags={IMAGE_TYPE_TAG: PHASE_IMAGE_TYPE}, is_regex=True,
            order=0)
            for i in range(epi_number))
    if run_regression:
        output_files.extend('epi_{}_smoothed_ts'.format(i)
                            for i in range(epi_number))
    else:
        output_files.extend('epi_{}_fix_dir'.format(i)
                            for i in range(epi_number))

    dct['add_substudy_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['add_param_specs'] = param_specs
    dct['__metaclass__'] = MultiStudyMetaClass
    return (MultiStudyMetaClass(name, (MultiBoldMixin,), dct), inputs,
            output_files)
