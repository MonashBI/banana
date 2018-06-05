from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.afni.preprocess import Volreg
from nipype.interfaces.fsl.utils import ImageMaths, ConvertXFM
from nianalysis.interfaces.fsl import (FSLFIX, FSLFixTraining,
                                       SignalRegression)
from arcana.dataset import DatasetSpec, FieldSpec
from arcana.study.base import StudyMetaClass
from nianalysis.requirement import (fsl5_req, afni_req, fix_req,
                                    fsl509_req, fsl510_req, ants2_req, c3d_req)
from nianalysis.citation import fsl_cite
from nianalysis.data_format import (
    nifti_gz_format, rdata_format, directory_format,
    zip_format, text_matrix_format, par_format, text_format, dicom_format)
from nianalysis.interfaces.afni import Tproject
from nipype.interfaces.utility import Merge as NiPypeMerge
import os
from nipype.interfaces.utility.base import IdentityInterface
from arcana.option import OptionSpec
from nianalysis.study.mri.epi import EPIStudy
from nipype.interfaces.ants.resampling import ApplyTransforms
from nianalysis.study.mri.structural.t1 import T1Study
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from arcana.dataset import DatasetMatch
from nipype.interfaces.afni.preprocess import BlurToFWHM
from nianalysis.interfaces.custom.fmri import PrepareFIX
from nianalysis.interfaces.c3d import ANTs2FSLMatrixConversion
import logging

logger = logging.getLogger('nianalysis')


atlas_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'atlases'))

IMAGE_TYPE_TAG = ('0008', '0008')
PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']
MAG_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']


class FunctionalMRIStudy(EPIStudy, metaclass=StudyMetaClass):

    add_option_specs = [
        OptionSpec('component_threshold', 20),
        OptionSpec('motion_reg', True),
        OptionSpec('highpass', 0.01),
        OptionSpec('brain_thresh_percent', 5),
        OptionSpec('MNI_template',
                   os.path.join(atlas_path, 'MNI152_T1_2mm.nii.gz')),
        OptionSpec('MNI_template_mask', os.path.join(
            atlas_path, 'MNI152_T1_2mm_brain_mask.nii.gz')),
        OptionSpec('linear_reg_method', 'ants'),
        OptionSpec('group_ica_components', 15)]

    add_data_specs = [
        DatasetSpec('train_data', rdata_format, 'fix_training_pipeline',
                    frequency='per_project'),
        DatasetSpec('labelled_components', text_format,
                    'fix_classification_pipeline'),
        DatasetSpec('cleaned_file', nifti_gz_format,
                    'fix_regression_pipeline'),
        DatasetSpec('filtered_data', nifti_gz_format,
                    'rsfMRI_filtering_pipeline'),
        DatasetSpec('mc_par', par_format, 'rsfMRI_filtering_pipeline'),
        DatasetSpec('melodic_ica', zip_format,
                    'single_subject_melodic_pipeline'),
        DatasetSpec('fix_dir', zip_format, 'fix_preparation_pipeline'),
        DatasetSpec('group_melodic', directory_format,
                    'group_melodic_pipeline', frequency='per_visit'),
        DatasetSpec('normalized_ts', nifti_gz_format,
                    'timeseries_normalization_to_atlas_pipeline'),
        DatasetSpec('smoothed_ts', nifti_gz_format,
                    'timeseries_spatial_smoothing_pipeline')]

    def rsfMRI_filtering_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='rsfMRI_filtering',
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('coreg_ref_brain', nifti_gz_format),
                    FieldSpec('tr', float)],
            outputs=[DatasetSpec('filtered_data', nifti_gz_format),
                     DatasetSpec('mc_par', par_format)],
            desc=("Spatial and temporal rsfMRI filtering"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        afni_mc = pipeline.create_node(Volreg(), name='AFNI_MC', wall_time=5,
                                       requirements=[afni_req])
        afni_mc.inputs.zpad = 1
        afni_mc.inputs.out_file = 'rsfmri_mc.nii.gz'
        afni_mc.inputs.oned_file = 'prefiltered_func_data_mcf.par'
        pipeline.connect_input('preproc', afni_mc, 'in_file')

        filt = pipeline.create_node(Tproject(), name='Tproject', wall_time=5,
                                    requirements=[afni_req])
        filt.inputs.stopband = (0, 0.01)
        filt.inputs.polort = 3
        filt.inputs.blur = 3
        filt.inputs.out_file = 'filtered_func_data.nii.gz'
        pipeline.connect_input('tr', filt, 'delta_t')
        pipeline.connect(afni_mc, 'out_file', filt, 'in_file')
        pipeline.connect_input('brain_mask', filt, 'mask')

        meanfunc = pipeline.create_node(
            ImageMaths(op_string='-Tmean', suffix='_mean'), name='meanfunc',
            wall_time=5, requirements=[fsl5_req])
        pipeline.connect(afni_mc, 'out_file', meanfunc, 'in_file')

        add_mean = pipeline.create_node(
            ImageMaths(op_string='-add'), name='add_mean', wall_time=5,
            requirements=[fsl5_req])
        pipeline.connect(filt, 'out_file', add_mean, 'in_file')
        pipeline.connect(meanfunc, 'out_file', add_mean, 'in_file2')

        pipeline.connect_output('filtered_data', add_mean, 'out_file')
        pipeline.connect_output('mc_par', afni_mc, 'oned_file')

        return pipeline

    def single_subject_melodic_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='MelodicL1',
            inputs=[DatasetSpec('filtered_data', nifti_gz_format),
                    FieldSpec('tr', float),
                    DatasetSpec('brain_mask', nifti_gz_format)],
            outputs=[DatasetSpec('melodic_ica', directory_format)],
            desc=("Single subject ICA analysis using FSL MELODIC."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        mel = pipeline.create_node(MELODIC(), name='melodic_L1', wall_time=15,
                                   requirements=[fsl5_req])
        mel.inputs.no_bet = True
        pipeline.connect_input('brain_mask', mel, 'mask')
        mel.inputs.bg_threshold = pipeline.option('brain_thresh_percent')
        mel.inputs.report = True
        mel.inputs.out_stats = True
        mel.inputs.mm_thresh = 0.5
        mel.inputs.out_dir = 'melodic_ica'
        pipeline.connect_input('tr', mel, 'tr_sec')
        pipeline.connect_input('filtered_data', mel, 'in_files')

        pipeline.connect_output('melodic_ica', mel, 'out_dir')

        return pipeline

    def fix_preparation_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='prepare_fix',
            inputs=[DatasetSpec('melodic_ica', directory_format),
                    DatasetSpec('filtered_data', nifti_gz_format),
                    DatasetSpec('coreg_to_atlas_mat', text_matrix_format),
                    DatasetSpec('coreg_matrix', text_matrix_format),
                    DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('brain', nifti_gz_format),
                    DatasetSpec('coreg_ref_brain', nifti_gz_format),
                    DatasetSpec('mc_par', par_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('primary', nifti_gz_format)],
            outputs=[DatasetSpec('fix_dir', directory_format)],
            desc=("Pipeline to create the right folder structure before "
                  "running FIX"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        struct_ants2fsl = pipeline.create_node(
            ANTs2FSLMatrixConversion(), name='struct_ants2fsl',
            requirements=[c3d_req])
        struct_ants2fsl.inputs.ras2fsl = True
        struct_ants2fsl.inputs.reference_file = pipeline.option('MNI_template')
        pipeline.connect_input('coreg_to_atlas_mat', struct_ants2fsl,
                               'itk_file')
        pipeline.connect_input('coreg_ref_brain', struct_ants2fsl,
                               'source_file')
        epi_ants2fsl = pipeline.create_node(
            ANTs2FSLMatrixConversion(), name='epi_ants2fsl',
            requirements=[c3d_req])
        epi_ants2fsl.inputs.ras2fsl = True
        pipeline.connect_input('brain', epi_ants2fsl,
                               'source_file')
        pipeline.connect_input('coreg_matrix', epi_ants2fsl,
                               'itk_file')
        pipeline.connect_input('coreg_ref_brain', epi_ants2fsl,
                               'reference_file')

        MNI2t1 = pipeline.create_node(ConvertXFM(), name='MNI2t1', wall_time=5,
                                      requirements=[fsl509_req])
        MNI2t1.inputs.invert_xfm = True
        MNI2t1.inputs.out_file = 'MNI2T1.mat'
        pipeline.connect(struct_ants2fsl, 'fsl_matrix', MNI2t1, 'in_file')

        struct2epi = pipeline.create_node(
            ConvertXFM(), name='struct2epi', wall_time=5,
            requirements=[fsl509_req])
        struct2epi.inputs.invert_xfm = True
        struct2epi.inputs.out_file = 'struct2epi.mat'
        pipeline.connect(epi_ants2fsl, 'fsl_matrix', struct2epi, 'in_file')

        meanfunc = pipeline.create_node(
            ImageMaths(op_string='-Tmean', suffix='_mean'), name='meanfunc',
            wall_time=5, requirements=[fsl509_req])
        pipeline.connect_input('primary', meanfunc, 'in_file')

        prep_fix = pipeline.create_node(PrepareFIX(), name='prep_fix')
        pipeline.connect_input('melodic_ica', prep_fix, 'melodic_dir')
        pipeline.connect_input('coreg_ref_brain', prep_fix, 't1_brain')
        pipeline.connect_input('mc_par', prep_fix, 'mc_par')
        pipeline.connect_input('brain_mask', prep_fix, 'epi_brain_mask')
        pipeline.connect_input('preproc', prep_fix, 'epi_preproc')
        pipeline.connect_input('filtered_data', prep_fix, 'filtered_epi')
        pipeline.connect(epi_ants2fsl, 'fsl_matrix', prep_fix, 'epi2t1_mat')
        pipeline.connect(struct_ants2fsl, 'fsl_matrix', prep_fix,
                         't12MNI_mat')
        pipeline.connect(MNI2t1, 'out_file', prep_fix, 'MNI2t1_mat')
        pipeline.connect(struct2epi, 'out_file', prep_fix, 't12epi_mat')
        pipeline.connect(meanfunc, 'out_file', prep_fix, 'epi_mean')

        pipeline.connect_output('fix_dir', prep_fix, 'fix_dir')

        return pipeline

    def fix_training_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='training_fix',
            inputs=[DatasetSpec('fix_dir', directory_format)],
            outputs=[DatasetSpec('train_data', rdata_format)],
            desc=("Pipeline to create the training set for FIX given a group "
                  "of subjects with the hand_label_noise.txt file within "
                  "their fix_dir."),
            version=1,
            citations=[fsl_cite],
            **kwargs)
#         labeled_sub = pipeline.create_join_subjects_node(
#             CheckLabelFile(), joinfield='in_list', name='labeled_subjects')
#         pipeline.connect_input('fix_dir', labeled_sub, 'in_list')
        merge_visits = pipeline.create_join_visits_node(
            IdentityInterface(['list_dir']), joinfield=['list_dir'],
            name='merge_visits')
        merge_subjects = pipeline.create_join_subjects_node(
            NiPypeMerge(1), joinfield=['in1'], name='merge_subjects')
        merge_subjects.inputs.ravel_inputs = True
        fix_training = pipeline.create_node(
            FSLFixTraining(), name='fix_training',
            wall_time=240, requirements=[fix_req])
        fix_training.inputs.outname = 'FIX_training_set'
        fix_training.inputs.training = True
        pipeline.connect_input('fix_dir', merge_visits, 'list_dir')
        pipeline.connect(merge_visits, 'list_dir', merge_subjects, 'in1')
        pipeline.connect(merge_subjects, 'out', fix_training, 'list_dir')

        pipeline.connect_output('train_data', fix_training, 'training_set')

        return pipeline

    def fix_classification_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='fix_classification',
            inputs=[DatasetSpec('train_data', rdata_format,
                                frequency='per_project'),
                    DatasetSpec('fix_dir', directory_format)],
            outputs=[DatasetSpec('labelled_components', text_format)],
            desc=("Automatic classification of noisy components from the "
                  "rsfMRI data using fsl FIX."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        fix = pipeline.create_node(FSLFIX(), name="fix", wall_time=30,
                                   requirements=[fsl509_req, fix_req])
        pipeline.connect_input("fix_dir", fix, "feat_dir")
        pipeline.connect_input("train_data", fix, "train_data")
        fix.inputs.component_threshold = pipeline.option(
            'component_threshold')
        fix.inputs.motion_reg = pipeline.option('motion_reg')
        fix.inputs.classification = True

        pipeline.connect_output('labelled_components', fix, 'label_file')

        return pipeline

    def fix_regression_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='signal_regression',
            inputs=[DatasetSpec('fix_dir', directory_format),
                    DatasetSpec('labelled_components', text_format)],
            outputs=[DatasetSpec('cleaned_file', nifti_gz_format)],
            desc=("Regression of the noisy components from the rsfMRI data "
                  "using a python implementation equivalent to that in FIX."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        signal_reg = pipeline.create_node(
            SignalRegression(), name="signal_reg", wall_time=30,
            requirements=[fsl509_req, fix_req])
        pipeline.connect_input("fix_dir", signal_reg, "fix_dir")
        pipeline.connect_input("labelled_components", signal_reg,
                               "labelled_components")
        signal_reg.inputs.motion_regression = pipeline.option('motion_reg')
        signal_reg.inputs.highpass = pipeline.option('highpass')

        pipeline.connect_output('cleaned_file', signal_reg, 'output')

        return pipeline

    def timeseries_normalization_to_atlas_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='timeseries_normalization_to_atlas_pipeline',
            inputs=[DatasetSpec('cleaned_file', nifti_gz_format),
                    DatasetSpec('coreg_to_atlas_warp', nifti_gz_format),
                    DatasetSpec('coreg_to_atlas_mat', text_matrix_format),
                    DatasetSpec('coreg_matrix', text_matrix_format)],
            outputs=[DatasetSpec('normalized_ts', nifti_gz_format)],
            desc=("Apply ANTs transformation to the fmri filtered file to "
                  "normalize it to MNI 2mm."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        merge_trans = pipeline.create_node(
            NiPypeMerge(3), name='merge_transforms', wall_time=1)
        pipeline.connect_input('coreg_to_atlas_warp', merge_trans, 'in1')
        pipeline.connect_input('coreg_to_atlas_mat', merge_trans, 'in2')
        pipeline.connect_input('coreg_matrix', merge_trans, 'in3')

        apply_trans = pipeline.create_node(
            ApplyTransforms(), name='ApplyTransform', wall_time=7,
            memory=24000, requirements=[ants2_req])
        ref_brain = pipeline.option('MNI_template')
        apply_trans.inputs.reference_image = ref_brain
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('cleaned_file', apply_trans, 'input_image')

        pipeline.connect_output('normalized_ts', apply_trans, 'output_image')

        return pipeline

    def timeseries_spatial_smoothing_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='timeseries_spatial_smoothing_pipeline',
            inputs=[DatasetSpec('normalized_ts', nifti_gz_format)],
            outputs=[DatasetSpec('smoothed_ts', nifti_gz_format)],
            desc=("Spatial smoothing of the normalized fmri file"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        smooth = pipeline.create_node(BlurToFWHM(), name='3dBlurToFWHM',
                                      wall_time=5, requirements=[afni_req])
        smooth.inputs.fwhm = 5
        smooth.inputs.out_file = 'smoothed_ts.nii.gz'
        smooth.inputs.mask = pipeline.option('MNI_template_mask')
        pipeline.connect_input('normalized_ts', smooth, 'in_file')

        pipeline.connect_output('smoothed_ts', smooth, 'out_file')

        return pipeline

    def group_melodic_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='group_melodic',
            inputs=[DatasetSpec('smoothed_ts', nifti_gz_format),
                    FieldSpec('tr', float)],
            outputs=[DatasetSpec('group_melodic', directory_format)],
            desc=("Group ICA"),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        gica = pipeline.create_join_subjects_node(
            MELODIC(), joinfield=['in_files'], name='gica',
            requirements=[fsl510_req], wall_time=7200)
        gica.inputs.no_bet = True
        gica.inputs.bg_threshold = pipeline.option('brain_thresh_percent')
        gica.inputs.bg_image = pipeline.option('MNI_template')
        gica.inputs.dim = pipeline.option('group_ica_components')
        gica.inputs.report = True
        gica.inputs.out_stats = True
        gica.inputs.mm_thresh = 0.5
        gica.inputs.sep_vn = True
        gica.inputs.mask = pipeline.option('MNI_template_mask')
        gica.inputs.out_dir = 'group_melodic.ica'
        pipeline.connect_input('smoothed_file', gica, 'in_files')
        pipeline.connect_input('tr', gica, 'tr_sec')

        pipeline.connect_output('group_melodic', gica, 'out_dir')

        return pipeline


def create_fmri_study_class(name, t1, epis, epi_number, fm_mag=None,
                            fm_phase=None, training_set=None):

    inputs = []
    dct = {}
    data_specs = []
    option_specs = []
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
    inputs.append(DatasetMatch('t1_primary', dicom_format, t1))
    epi_refspec = ref_spec.copy()
    epi_refspec.update({'t1_wm_seg': 'coreg_ref_wmseg',
                        't1_preproc': 'coreg_ref_preproc'})
    study_specs.extend(SubStudySpec('epi_{}'.format(i), FunctionalMRIStudy,
                                    ref_spec)
                       for i in range(epi_number))

    for i in range(epi_number):
        inputs.append(DatasetMatch('epi_{}_primary'.format(i),
                                   dicom_format, epis, order=i))

#     inputs.extend(
#         DatasetMatch('epi_{}_primary'.format(i), dicom_format, epi_scan)
#         for i, epi_scan in enumerate(epis))
    if distortion_correction:
        inputs.extend(DatasetMatch(
            'epi_{}_field_map_mag'.format(i), dicom_format, fm_mag,
            dicom_tags={IMAGE_TYPE_TAG: MAG_IMAGE_TYPE})
            for i in range(epi_number))
        inputs.extend(DatasetMatch(
            'epi_{}_field_map_phase'.format(i), dicom_format, fm_phase,
            dicom_tags={IMAGE_TYPE_TAG: PHASE_IMAGE_TYPE})
            for i in range(epi_number))
    if training_set is not None:
        inputs.extend(DatasetMatch('epi_{}_train_data'.format(i),
                                   rdata_format, training_set)
                      for i in range(epi_number))
        output_files.extend('epi_{}_smoothed_ts'.format(i)
                            for i in range(epi_number))
    else:
        output_files.extend('epi_{}_melodic_ica'.format(i)
                            for i in range(epi_number))

    dct['add_sub_study_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['add_option_specs'] = option_specs
    return MultiStudyMetaClass(name, (MultiStudy,), dct), inputs, output_files
