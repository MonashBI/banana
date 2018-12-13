from nipype.interfaces.fsl.model import MELODIC
from nipype.interfaces.afni.preprocess import Volreg
from nipype.interfaces.fsl.utils import ImageMaths, ConvertXFM
from banana.interfaces.fsl import (FSLFIX, FSLFixTraining,
                                       SignalRegression, PrepareFIXTraining)
from arcana.data import FilesetSpec
from arcana.study.base import StudyMetaClass
from banana.requirement import (
    afni_req, fix_req, fsl_req, ants_req, c3d_req)
from banana.citation import fsl_cite
from banana.file_format import (
    nifti_gz_format, rfile_format, directory_format,
    zip_format, par_format, text_format, dicom_format)
from banana.interfaces.afni import Tproject
from nipype.interfaces.utility import Merge as NiPypeMerge
import os.path as op
from nipype.interfaces.utility.base import IdentityInterface
from arcana.study import ParameterSpec, SwitchSpec
from banana.study.mri.epi import EpiStudy
from nipype.interfaces.ants.resampling import ApplyTransforms
from banana.study.mri.t1 import T1Study
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from arcana.data import FilesetSelector
from nipype.interfaces.afni.preprocess import BlurToFWHM
from banana.interfaces.custom.fmri import PrepareFIX
from banana.interfaces.c3d import ANTs2FSLMatrixConversion
import logging
from arcana.exceptions import ArcanaNameError

logger = logging.getLogger('banana')


atlas_path = op.abspath(
    op.join(op.dirname(__file__), '..', '..', '..', 'atlases'))

IMAGE_TYPE_TAG = ('0008', '0008')
PHASE_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'P', 'ND']
MAG_IMAGE_TYPE = ['ORIGINAL', 'PRIMARY', 'M', 'ND', 'NORM']


class FmriStudy(EpiStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('train_data', rfile_format, optional=True,
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
        ParameterSpec('component_threshold', 20),
        ParameterSpec('motion_reg', True),
        ParameterSpec('highpass', 0.01),
        ParameterSpec('brain_thresh_percent', 5),
        ParameterSpec('MNI_template', op.join(atlas_path,
                                              'MNI152_T1_2mm.nii.gz')),
        ParameterSpec('MNI_template_mask', op.join(
            atlas_path, 'MNI152_T1_2mm_brain_mask.nii.gz')),
        SwitchSpec('linear_reg_method', 'ants',
                   ('flirt', 'spm', 'ants', 'epireg')),
        ParameterSpec('group_ica_components', 15)]

    def rsfMRI_filtering_pipeline(self, **kwargs):

        
#             inputs=[FilesetSpec('preproc', nifti_gz_format),
#                     FilesetSpec('brain_mask', nifti_gz_format),
#                     FilesetSpec('coreg_ref_brain', nifti_gz_format),
#                     FieldSpec('tr', float)],
#             outputs=[FilesetSpec('filtered_data', nifti_gz_format),
#                      FilesetSpec('mc_par', par_format)],
        
        pipeline = self.new_pipeline(
            name='rsfMRI_filtering',
            desc=("Spatial and temporal rsfMRI filtering"),
            citations=[fsl_cite],
            **kwargs)

        afni_mc = pipeline.add(
            'AFNI_MC',
            Volreg(),
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])
        afni_mc.inputs.zpad = 1
        afni_mc.inputs.out_file = 'rsfmri_mc.nii.gz'
        afni_mc.inputs.oned_file = 'prefiltered_func_data_mcf.par'
        pipeline.connect_input('preproc', afni_mc, 'in_file')

        filt = pipeline.add(
            'Tproject',
            Tproject(),
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])
        filt.inputs.stopband = (0, 0.01)
        filt.inputs.polort = 3
        filt.inputs.blur = 3
        filt.inputs.out_file = 'filtered_func_data.nii.gz'
        pipeline.connect_input('tr', filt, 'delta_t')
        pipeline.connect(afni_mc, 'out_file', filt, 'in_file')
        pipeline.connect_input('brain_mask', filt, 'mask')

        meanfunc = pipeline.add(
            'meanfunc',
            ImageMaths(op_string='-Tmean', suffix='_mean'),
            wall_time=5,
            requirements=[fsl_req.v('5.0.10')])
        pipeline.connect(afni_mc, 'out_file', meanfunc, 'in_file')

        add_mean = pipeline.add(
            'add_mean',
            ImageMaths(op_string='-add'),
            wall_time=5,
            requirements=[fsl_req.v('5.0.10')])
        pipeline.connect(filt, 'out_file', add_mean, 'in_file')
        pipeline.connect(meanfunc, 'out_file', add_mean, 'in_file2')

        pipeline.connect_output('filtered_data', add_mean, 'out_file')
        pipeline.connect_output('mc_par', afni_mc, 'oned_file')

        return pipeline

    def single_subject_melodic_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('filtered_data', nifti_gz_format),
#                     FieldSpec('tr', float),
#                     FilesetSpec('brain_mask', nifti_gz_format)],
#             outputs=[FilesetSpec('melodic_ica', directory_format)],

        pipeline = self.new_pipeline(
            name='MelodicL1',
            desc=("Single subject ICA analysis using FSL MELODIC."),
            citations=[fsl_cite],
            **kwargs)

        mel = pipeline.add(
            'melodic_L1',
            MELODIC(),
            wall_time=15,
            requirements=[fsl_req.v('5.0.10')])
        mel.inputs.no_bet = True
        pipeline.connect_input('brain_mask', mel, 'mask')
        mel.inputs.bg_threshold = self.parameter('brain_thresh_percent')
        mel.inputs.report = True
        mel.inputs.out_stats = True
        mel.inputs.mm_thresh = 0.5
        mel.inputs.out_dir = 'melodic_ica'
        pipeline.connect_input('tr', mel, 'tr_sec')
        pipeline.connect_input('filtered_data', mel, 'in_files')

        pipeline.connect_output('melodic_ica', mel, 'out_dir')

        return pipeline

    def fix_preparation_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('melodic_ica', directory_format),
#                     FilesetSpec('filtered_data', nifti_gz_format),
#                     FilesetSpec('coreg_to_atlas_mat', text_matrix_format),
#                     FilesetSpec('coreg_matrix', text_matrix_format),
#                     FilesetSpec('preproc', nifti_gz_format),
#                     FilesetSpec('brain', nifti_gz_format),
#                     FilesetSpec('coreg_ref_brain', nifti_gz_format),
#                     FilesetSpec('mc_par', par_format),
#                     FilesetSpec('brain_mask', nifti_gz_format)],
#             outputs=[FilesetSpec('fix_dir', directory_format),
#                      FilesetSpec('hand_label_noise', text_format)],

        pipeline = self.new_pipeline(
            name='prepare_fix',
            desc=("Pipeline to create the right folder structure before "
                  "running FIX"),
            citations=[fsl_cite],
            **kwargs)

        struct_ants2fsl = pipeline.add(
            'struct_ants2fsl',
            ANTs2FSLMatrixConversion(),
            requirements=[c3d_req.v('1.1.0')])
        struct_ants2fsl.inputs.ras2fsl = True
        struct_ants2fsl.inputs.reference_file = self.parameter('MNI_template')
        pipeline.connect_input('coreg_to_atlas_mat', struct_ants2fsl,
                               'itk_file')
        pipeline.connect_input('coreg_ref_brain', struct_ants2fsl,
                               'source_file')
        epi_ants2fsl = pipeline.add(
            'epi_ants2fsl',
            ANTs2FSLMatrixConversion(),
            requirements=[c3d_req.v('1.1.0')])
        epi_ants2fsl.inputs.ras2fsl = True
        pipeline.connect_input('brain', epi_ants2fsl, 'source_file')
        pipeline.connect_input('coreg_matrix', epi_ants2fsl, 'itk_file')
        pipeline.connect_input('coreg_ref_brain', epi_ants2fsl,
                               'reference_file')

        MNI2t1 = pipeline.add(
            'MNI2t1',
            ConvertXFM(),
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])
        MNI2t1.inputs.invert_xfm = True
        pipeline.connect(struct_ants2fsl, 'fsl_matrix', MNI2t1, 'in_file')

        struct2epi = pipeline.add(
            'struct2epi',
            ConvertXFM(),
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])
        struct2epi.inputs.invert_xfm = True
        pipeline.connect(epi_ants2fsl, 'fsl_matrix', struct2epi, 'in_file')

        meanfunc = pipeline.add(
            'meanfunc',
            ImageMaths(op_string='-Tmean', suffix='_mean'),
            wall_time=5,
            requirements=[fsl_req.v('5.0.9')])
        pipeline.connect_input('preproc', meanfunc, 'in_file')

        prep_fix = pipeline.add(
            'prep_fix',
            PrepareFIX())
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
        pipeline.connect_output('hand_label_noise', prep_fix,
                                'hand_label_file')

        return pipeline

    def fix_classification_pipeline(self, **kwargs):


#             inputs=[DatasetSpec('train_data', rfile_format,
#                                 frequency='per_project'),
#                     DatasetSpec('fix_dir', directory_format)],
#             outputs=[DatasetSpec('labelled_components', text_format)],

        pipeline = self.create_pipeline(
            name='fix_classification',
            desc=("Automatic classification of noisy components from the "
                  "rsfMRI data using fsl FIX."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        fix = pipeline.add(
            "fix",
            FSLFIX(),
            wall_time=30,
            requirements=[fsl_req.v('5.0.9'), fix_req.v('1.0')])
        pipeline.connect_input("fix_dir", fix, "feat_dir")
        pipeline.connect_input("train_data", fix, "train_data")
        fix.inputs.component_threshold = self.parameter(
            'component_threshold')
        fix.inputs.motion_reg = self.parameter('motion_reg')
        fix.inputs.classification = True

        pipeline.connect_output('labelled_components', fix, 'label_file')

        return pipeline

#     def fix_training_pipeline(self, **kwargs):
# 
#         inputs = []
#         sub_study_names = []
#         for sub_study_spec in self.sub_study_specs():
#             try:
#                 spec = self.data_spec(sub_study_spec.inverse_map('fix_dir'))
#                 spec._format = directory_format
#                 inputs.append(spec)
#                 inputs.append(
#                     self.data_spec(sub_study_spec.inverse_map(
#                         'hand_label_noise')))
#                 sub_study_names.append(sub_study_spec.name)
#             except ArcanaNameError:
#                 continue  # Sub study doesn't have fix dir
# 
#         pipeline = self.new_pipeline(
#             name='training_fix',
#             inputs=inputs,
#             outputs=[FilesetSpec('train_data', rfile_format)],
#             desc=("Pipeline to create the training set for FIX given a group "
#                   "of subjects with the hand_label_noise.txt file within "
#                   "their fix_dir."),
#             version=1,
#             citations=[fsl_cite],
#             **kwargs)
# 
#         num_fix_dirs = len(sub_study_names)
#         merge_fix_dirs = pipeline.create_node(NiPypeMerge(num_fix_dirs),
#                                               name='merge_fix_dirs')
#         merge_label_files = pipeline.create_node(NiPypeMerge(num_fix_dirs),
#                                                  name='merge_label_files')
#         for i, sub_study_name in enumerate(sub_study_names, start=1):
#             spec = self.sub_study_spec(sub_study_name)
#             pipeline.connect_input(
#                 spec.inverse_map('fix_dir'), merge_fix_dirs, 'in{}'.format(i))
#             pipeline.connect_input(
#                 spec.inverse_map('hand_label_noise'), merge_label_files,
#                 'in{}'.format(i))
# 
#         merge_visits = pipeline.create_join_visits_node(
#             IdentityInterface(['list_dir', 'list_label_files']),
#             joinfield=['list_dir', 'list_label_files'], name='merge_visits')
#         merge_subjects = pipeline.create_join_subjects_node(
#             NiPypeMerge(2), joinfield=['in1', 'in2'], name='merge_subjects')
#         merge_subjects.inputs.ravel_inputs = True
# 
#         prepare_training = pipeline.create_node(PrepareFIXTraining(),
#                                                 name='prepare_training')
#         prepare_training.inputs.epi_number = num_fix_dirs
#         pipeline.connect(merge_fix_dirs, 'out', merge_visits, 'list_dir')
#         pipeline.connect(merge_visits, 'list_dir', merge_subjects, 'in1')
#         pipeline.connect(merge_label_files, 'out', merge_visits,
#                          'list_label_files')
#         pipeline.connect(merge_visits, 'list_label_files', merge_subjects,
#                          'in2')
#         pipeline.connect(merge_subjects, 'out', prepare_training,
#                          'inputs_list')
# 
#         fix_training = pipeline.create_node(
#             FSLFixTraining(), name='fix_training',
#             wall_time=240, requirements=[fix_req.v('1.0')])
#         fix_training.inputs.outname = 'FIX_training_set'
#         fix_training.inputs.training = True
#         pipeline.connect(prepare_training, 'prepared_dirs', fix_training,
#                          'list_dir')
# 
#         pipeline.connect_output('train_data', fix_training, 'training_set')
# 
#         return pipeline
# 
#     def fix_classification_pipeline(self, **kwargs):
# 
# 
# #             inputs=[FilesetSpec('train_data', rfile_format,
# #                                 frequency='per_study'),
# #                     FilesetSpec('fix_dir', directory_format)],
# #             outputs=[FilesetSpec('labelled_components', text_format)],
# 
#         pipeline = self.new_pipeline(
#             name='fix_classification',
#             desc=("Automatic classification of noisy components from the "
#                   "rsfMRI data using fsl FIX."),
#             citations=[fsl_cite],
#             **kwargs)
# 
#         fix = pipeline.add(
#             "fix",
#             FSLFIX(),
#             wall_time=30,
#             requirements=[fsl_req.v('5.0.9'), fix_req.v('1.0')])
#         pipeline.connect_input("fix_dir", fix, "feat_dir")
#         pipeline.connect_input("train_data", fix, "train_data")
#         fix.inputs.component_threshold = self.parameter(
#             'component_threshold')
#         fix.inputs.motion_reg = self.parameter('motion_reg')
#         fix.inputs.classification = True
# 
#         pipeline.connect_output('labelled_components', fix, 'label_file')
# 
#         return pipeline

    def fix_regression_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('fix_dir', directory_format),
#                     FilesetSpec('labelled_components', text_format)],
#             outputs=[FilesetSpec('cleaned_file', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='signal_regression',
            desc=("Regression of the noisy components from the rsfMRI data "
                  "using a python implementation equivalent to that in FIX."),
            citations=[fsl_cite],
            **kwargs)

        signal_reg = pipeline.add(
            "signal_reg",
            SignalRegression(),
            wall_time=30,
            requirements=[fsl_req.v('5.0.9'), fix_req.v('1.0')])
        pipeline.connect_input("fix_dir", signal_reg, "fix_dir")
        pipeline.connect_input("labelled_components", signal_reg,
                               "labelled_components")
        signal_reg.inputs.motion_regression = self.parameter('motion_reg')
        signal_reg.inputs.highpass = self.parameter('highpass')

        pipeline.connect_output('cleaned_file', signal_reg, 'output')

        return pipeline

    def timeseries_normalization_to_atlas_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('cleaned_file', nifti_gz_format),
#                     FilesetSpec('coreg_to_atlas_warp', nifti_gz_format),
#                     FilesetSpec('coreg_to_atlas_mat', text_matrix_format),
#                     FilesetSpec('coreg_matrix', text_matrix_format)],
#             outputs=[FilesetSpec('normalized_ts', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='timeseries_normalization_to_atlas_pipeline',
            desc=("Apply ANTs transformation to the fmri filtered file to "
                  "normalize it to MNI 2mm."),
            citations=[fsl_cite],
            **kwargs)

        merge_trans = pipeline.add(
            'merge_transforms',
            NiPypeMerge(3),
            wall_time=1)
        pipeline.connect_input('coreg_to_atlas_warp', merge_trans, 'in1')
        pipeline.connect_input('coreg_to_atlas_mat', merge_trans, 'in2')
        pipeline.connect_input('coreg_matrix', merge_trans, 'in3')

        apply_trans = pipeline.add(
            'ApplyTransform',
            ApplyTransforms(),
            wall_time=7,
            mem_gb=24,
            requirements=[ants_req.v('2')])
        ref_brain = self.parameter('MNI_template')
        apply_trans.inputs.reference_image = ref_brain
        apply_trans.inputs.interpolation = 'Linear'
        apply_trans.inputs.input_image_type = 3
        pipeline.connect(merge_trans, 'out', apply_trans, 'transforms')
        pipeline.connect_input('cleaned_file', apply_trans, 'input_image')

        pipeline.connect_output('normalized_ts', apply_trans, 'output_image')

        return pipeline

    def smoothing_pipeline(self, **kwargs):


#             inputs=[FilesetSpec('normalized_ts', nifti_gz_format)],
#             outputs=[FilesetSpec('smoothed_ts', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='smoothing_pipeline',
            desc=("Spatial smoothing of the normalized fmri file"),
            citations=[fsl_cite],
            **kwargs)

        smooth = pipeline.add(
            '3dBlurToFWHM',
            BlurToFWHM(),
            wall_time=5,
            requirements=[afni_req.v('16.2.10')])
        smooth.inputs.fwhm = 5
        smooth.inputs.out_file = 'smoothed_ts.nii.gz'
        smooth.inputs.mask = self.parameter('MNI_template_mask')
        pipeline.connect_input('normalized_ts', smooth, 'in_file')

        pipeline.connect_output('smoothed_ts', smooth, 'out_file')

        return pipeline


class FmriMixin(MultiStudy, metaclass=MultiStudyMetaClass):

    add_data_specs = [
        FilesetSpec('train_data', rfile_format, 'fix_training_pipeline',
                    frequency='per_study'),
#         FilesetSpec('fmri_pre-processeing_results', directory_format,
#                     'gather_fmri_result_pipeline'),
        FilesetSpec('group_melodic', directory_format, 'group_melodic_pipeline')]

    def fix_training_pipeline(self, **kwargs):

        inputs = []
        sub_study_names = []
        for sub_study_spec in self.sub_study_specs():
            try:
                spec = self.data_spec(sub_study_spec.inverse_map('fix_dir'))
                spec._format = directory_format
                inputs.append(spec)
                inputs.append(
                    self.data_spec(sub_study_spec.inverse_map(
                        'hand_label_noise')))
                sub_study_names.append(sub_study_spec.name)
            except ArcanaNameError:
                continue  # Sub study doesn't have fix dir
            
        
#             inputs=inputs,
#             outputs=[FilesetSpec('train_data', rfile_format)],

        pipeline = self.new_pipeline(
            name='training_fix',
            desc=("Pipeline to create the training set for FIX given a group "
                  "of subjects with the hand_label_noise.txt file within "
                  "their fix_dir."),
            citations=[fsl_cite],
            **kwargs)

        num_fix_dirs = len(sub_study_names)
        merge_fix_dirs = pipeline.add(
            'merge_fix_dirs',
            NiPypeMerge(num_fix_dirs))
        merge_label_files = pipeline.add(
            'merge_label_files',
            NiPypeMerge(num_fix_dirs))
        for i, sub_study_name in enumerate(sub_study_names, start=1):
            spec = self.sub_study_spec(sub_study_name)
            pipeline.connect_input(
                spec.inverse_map('fix_dir'), merge_fix_dirs, 'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('hand_label_noise'), merge_label_files,
                'in{}'.format(i))

        merge_visits = pipeline.create_join_visits_node(
            IdentityInterface(['list_dir', 'list_label_files']),
            joinfield=['list_dir', 'list_label_files'], name='merge_visits')
        merge_subjects = pipeline.create_join_subjects_node(
            NiPypeMerge(2), joinfield=['in1', 'in2'], name='merge_subjects')
        merge_subjects.inputs.ravel_inputs = True

        prepare_training = pipeline.add(
            'prepare_training',
            PrepareFIXTraining())
        prepare_training.inputs.epi_number = num_fix_dirs
        pipeline.connect(merge_fix_dirs, 'out', merge_visits, 'list_dir')
        pipeline.connect(merge_visits, 'list_dir', merge_subjects, 'in1')
        pipeline.connect(merge_label_files, 'out', merge_visits,
                         'list_label_files')
        pipeline.connect(merge_visits, 'list_label_files', merge_subjects,
                         'in2')
        pipeline.connect(merge_subjects, 'out', prepare_training,
                         'inputs_list')

        fix_training = pipeline.add(
            'fix_training',
            FSLFixTraining(),
            wall_time=240,
            requirements=[fix_req.v('1.0')])
        fix_training.inputs.outname = 'FIX_training_set'
        fix_training.inputs.training = True
        pipeline.connect(prepare_training, 'prepared_dirs', fix_training,
                         'list_dir')

        pipeline.connect_output('train_data', fix_training, 'training_set')

        return pipeline
    
#     def gather_fmri_result_pipeline(self, **kwargs):
# 
#         inputs = []
#         sub_study_names = []
#         for sub_study_spec in self.sub_study_specs():
#             try:
#                 inputs.append(
#                     self.data_spec(sub_study_spec.inverse_map('smoothed_ts')))
#                 sub_study_names.append(sub_study_spec.name)
#             except ArcanaNameError:
#                 continue  # Sub study doesn't have fix dir
# 
#         pipeline = self.new_pipeline(
#             name='gather_fmri',
#             inputs=inputs,
#             outputs=[FilesetSpec('fmri_pre-processeing_results', directory_format)],
#             desc=("Pipeline to gather together all the pre-processed fMRI images"),
#             version=1,
#             citations=[fsl_cite],
#             **kwargs)
# 
#         merge_inputs = pipeline.create_node(NiPypeMerge(len(inputs)),
#                                             name='merge_inputs')
#         for i, sub_study_name in enumerate(sub_study_names, start=1):
#             spec = self.sub_study_spec(sub_study_name)
#             pipeline.connect_input(
#                 spec.inverse_map('smoothed_ts'), merge_inputs, 'in{}'.format(i))
# 
#         copy2dir = pipeline.add('copy2dir', CopyToDir())
#         pipeline.connect(merge_inputs, 'out', copy2dir, 'in_files')
# 
#         pipeline.connect_output('fmri_pre-processeing_results', copy2dir, 'out_dir')
#         return pipeline

    def group_melodic_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('smoothed_ts', nifti_gz_format),
#                     FieldSpec('tr', float)],
#             outputs=[FilesetSpec('group_melodic', directory_format)],

        pipeline = self.new_pipeline(
            name='group_melodic',
            desc=("Group ICA"),
            citations=[fsl_cite],
            **kwargs)
        gica = pipeline.create_join_subjects_node(
            MELODIC(), joinfield=['in_files'], name='gica',
            requirements=[fsl_req.v('5.0.10')],
            wall_time=7200)
        gica.inputs.no_bet = True
        gica.inputs.bg_threshold = self.parameter('brain_thresh_percent')
        gica.inputs.bg_image = self.parameter('MNI_template')
        gica.inputs.dim = self.parameter('group_ica_components')
        gica.inputs.report = True
        gica.inputs.out_stats = True
        gica.inputs.mm_thresh = 0.5
        gica.inputs.sep_vn = True
        gica.inputs.mask = self.parameter('MNI_template_mask')
        gica.inputs.out_dir = 'group_melodic.ica'
        pipeline.connect_input('smoothed_ts', gica, 'in_files')
        pipeline.connect_input('tr', gica, 'tr_sec')

        pipeline.connect_output('group_melodic', gica, 'out_dir')

        return pipeline


def create_fmri_study_class(name, t1, epis, epi_number, echo_spacing,
                            fm_mag=None, fm_phase=None, run_regression=False):

    inputs = []
    dct = {}
    data_specs = []
    parameter_specs = []
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
    inputs.append(FilesetSelector('t1_primary', dicom_format, t1, is_regex=True,
                               order=0))
    epi_refspec = ref_spec.copy()
    epi_refspec.update({'t1_wm_seg': 'coreg_ref_wmseg',
                        't1_preproc': 'coreg_ref_preproc',
                        'train_data': 'train_data'})
    study_specs.append(SubStudySpec('epi_0', FmriStudy, epi_refspec))
    if epi_number > 1:
        epi_refspec.update({'t1_wm_seg': 'coreg_ref_wmseg',
                            't1_preproc': 'coreg_ref_preproc',
                            'train_data': 'train_data',
                            'epi_0_coreg_to_atlas_warp': 'coreg_to_atlas_warp',
                            'epi_0_coreg_to_atlas_mat': 'coreg_to_atlas_mat'})
        study_specs.extend(SubStudySpec('epi_{}'.format(i), FmriStudy,
                                        epi_refspec)
                           for i in range(1, epi_number))

    study_specs.extend(SubStudySpec('epi_{}'.format(i), FmriStudy,
                                    epi_refspec)
                       for i in range(epi_number))

    for i in range(epi_number):
        inputs.append(FilesetSelector('epi_{}_primary'.format(i),
                                   dicom_format, epis, order=i, is_regex=True))
#     inputs.extend(FilesetSelector(
#         'epi_{}_hand_label_noise'.format(i), text_format,
#         'hand_label_noise_{}'.format(i+1))
#         for i in range(epi_number))
        parameter_specs.append(
            ParameterSpec('epi_{}_fugue_echo_spacing'.format(i), echo_spacing))

    if distortion_correction:
        inputs.extend(FilesetSelector(
            'epi_{}_field_map_mag'.format(i), dicom_format, fm_mag,
            dicom_tags={IMAGE_TYPE_TAG: MAG_IMAGE_TYPE}, is_regex=True,
            order=0)
            for i in range(epi_number))
        inputs.extend(FilesetSelector(
            'epi_{}_field_map_phase'.format(i), dicom_format, fm_phase,
            dicom_tags={IMAGE_TYPE_TAG: PHASE_IMAGE_TYPE}, is_regex=True,
            order=0)
            for i in range(epi_number))
    if run_regression:
        output_files.extend('epi_{}_smoothed_ts'.format(i)
                            for i in range(epi_number))
    else:
        output_files.extend('epi_{}_fix_dir'.format(i)
                            for i in range(epi_number))

    dct['add_sub_study_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['add_param_specs'] = parameter_specs
    dct['__metaclass__'] = MultiStudyMetaClass
    return (MultiStudyMetaClass(name, (FmriMixin,), dct), inputs,
            output_files)
