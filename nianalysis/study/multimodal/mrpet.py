from arcana.dataset import DatasetSpec, FieldSpec
from nianalysis.data_format import (
    nifti_gz_format, text_matrix_format, directory_format, text_format,
    png_format, dicom_format)
from nianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming, PlotMeanDisplacementRC,
    AffineMatAveraging, PetCorrectionFactor, FrameAlign2Reference,
    CreateMocoSeries, FixedBinning)
from nianalysis.citation import fsl_cite
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from nianalysis.study.mri.epi import EPIStudy
from nianalysis.study.mri.structural.t1 import T1Study
from nianalysis.study.mri.structural.t2 import T2Study
from nipype.interfaces.utility import Merge
from nianalysis.study.mri.structural.diffusion_coreg import DWIStudy
from nianalysis.requirement import fsl509_req
from arcana.exception import ArcanaNameError
from arcana.dataset import DatasetMatch
import logging
from nianalysis.study.pet.base import PETStudy
from nianalysis.interfaces.custom.pet import (
    CheckPetMCInputs, PetImageMotionCorrection, StaticPETImageGeneration,
    PETFovCropping)
from arcana.option import OptionSpec
import os
from arcana.interfaces.converters import Nii2Dicom
from arcana.interfaces.utils import CopyToDir, ListDir, dicom_fname_sort_key
from nipype.interfaces.fsl.preprocess import FLIRT
from arcana.study.base import StudyMetaClass


logger = logging.getLogger('Arcana')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../nianalysis',
                 'reference_data'))


class MotionReferenceT1Study(T1Study):

    __metaclass__ = StudyMetaClass

    def motion_mat_pipeline(self, **kwargs):
        pipeline = super(T1Study, self).motion_mat_pipeline(ref=True, **kwargs)
        return pipeline


class MotionReferenceT2Study(T2Study):

    __metaclass__ = StudyMetaClass

    def motion_mat_pipeline(self, **kwargs):
        pipeline = super(T2Study, self).motion_mat_pipeline(ref=True, **kwargs)
        return pipeline


class MotionDetectionMixin(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    add_sub_study_specs = [
        SubStudySpec('pet_mc', PETStudy, {
            'pet_data_dir': 'pet_data_dir',
            'pet_data_reconstructed': 'pet_recon_dir',
            'pet_data_prepared': 'pet_recon_dir_prepared',
            'pet_start_time': 'pet_start_time',
            'pet_end_time': 'pet_end_time',
            'pet_duration': 'pet_duration'})]

    add_data_specs = [
        DatasetSpec('pet_data_dir', directory_format, optional=True),
        DatasetSpec('pet_data_reconstructed', directory_format, optional=True),
        DatasetSpec('pet_data_prepared', directory_format,
                    'prepare_pet_pipeline'),
        DatasetSpec('static_motion_correction_results', directory_format,
                    'static_motion_correction_pipeline'),
        DatasetSpec('dynamic_motion_correction_results', directory_format,
                    'dynamic_motion_correction_pipeline'),
        DatasetSpec('umap', dicom_format, optional=True),
        DatasetSpec('mean_displacement', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_rc', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mean_displacement_consecutive', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('mats4average', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('start_times', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('motion_par_rc', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('motion_par', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('offset_indexes', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('severe_motion_detection_report', text_format,
                    'mean_displacement_pipeline'),
        DatasetSpec('frame_start_times', text_format,
                    'motion_framing_pipeline'),
        DatasetSpec('frame_vol_numbers', text_format,
                    'motion_framing_pipeline'),
        DatasetSpec('timestamps', directory_format,
                    'motion_framing_pipeline'),
        DatasetSpec('mean_displacement_plot', png_format,
                    'plot_mean_displacement_pipeline'),
        DatasetSpec('average_mats', directory_format,
                    'frame_mean_transformation_mats_pipeline'),
        DatasetSpec('correction_factors', text_format,
                    'pet_correction_factors_pipeline'),
        DatasetSpec('umaps_align2ref', directory_format,
                    'static_frame2ref_alignment_pipeline'),
        DatasetSpec('umap_aligned_dicoms', directory_format,
                    'nifti2dcm_conversion_pipeline'),
        DatasetSpec('static_frame2reference_mats', directory_format,
                    'static_frame2ref_alignment_pipeline'),
        DatasetSpec('dynamic_frame2reference_mats', directory_format,
                    'dynamic_frame2ref_alignment_pipeline'),
        DatasetSpec('motion_detection_output', directory_format,
                    'gather_outputs_pipeline'),
        DatasetSpec('moco_series', directory_format,
                    'create_moco_series_pipeline'),
        DatasetSpec('fixed_binning_mats', directory_format,
                    'fixed_binning_pipeline'),
        FieldSpec('pet_duration', dtype=int,
                  pipeline_name='pet_header_info_extraction_pipeline'),
        FieldSpec('pet_end_time', dtype=str,
                  pipeline_name='pet_header_info_extraction_pipeline'),
        FieldSpec('pet_start_time', dtype=str,
                  pipeline_name='pet_header_info_extraction_pipeline')]

    add_option_specs = [OptionSpec('framing_th', 2.0),
                        OptionSpec('framing_temporal_th', 30.0),
                        OptionSpec('md_framing', True),
                        OptionSpec('align_pct', False),
                        OptionSpec('align_fixed_binning', False),
                        OptionSpec('moco_template', os.path.join(
                            template_path, 'moco_template.IMA')),
                        OptionSpec('fixed_binning_n_frames', 'all'),
                        OptionSpec('fixed_binning_pet_offset', 0),
                        OptionSpec('fixed_binning_bin_len', 60)]

    def mean_displacement_pipeline(self, **kwargs):
        inputs = [DatasetSpec('ref_brain', nifti_gz_format)]
        sub_study_names = []
        input_names = []
        for sub_study_spec in self.sub_study_specs():
            try:
                inputs.append(
                    self.data_spec(sub_study_spec.inverse_map('motion_mats')))
                inputs.append(self.data_spec(sub_study_spec.inverse_map('tr')))
                inputs.append(
                    self.data_spec(sub_study_spec.inverse_map('start_time')))
                inputs.append(
                    self.data_spec(sub_study_spec.inverse_map(
                        'real_duration')))
                input_names.append(
                    self.spec(sub_study_spec.inverse_map(
                        'primary')).pattern)
                sub_study_names.append(sub_study_spec.name)
            except ArcanaNameError:
                continue  # Sub study doesn't have motion mat

        pipeline = self.create_pipeline(
            name='mean_displacement_calculation',
            inputs=inputs,
            outputs=[DatasetSpec('mean_displacement', text_format),
                     DatasetSpec('mean_displacement_rc', text_format),
                     DatasetSpec('mean_displacement_consecutive', text_format),
                     DatasetSpec('start_times', text_format),
                     DatasetSpec('motion_par_rc', text_format),
                     DatasetSpec('motion_par', text_format),
                     DatasetSpec('offset_indexes', text_format),
                     DatasetSpec('mats4average', text_format),
                     DatasetSpec('severe_motion_detection_report',
                                 text_format)],
            desc=("Calculate the mean displacement between each motion"
                  " matrix and a reference."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        num_motion_mats = len(sub_study_names)
        merge_motion_mats = pipeline.create_node(Merge(num_motion_mats),
                                                 name='merge_motion_mats')
        merge_tr = pipeline.create_node(Merge(num_motion_mats),
                                        name='merge_tr')
        merge_start_time = pipeline.create_node(Merge(num_motion_mats),
                                                name='merge_start_time')
        merge_real_duration = pipeline.create_node(Merge(num_motion_mats),
                                                   name='merge_real_duration')

        for i, sub_study_name in enumerate(sub_study_names, start=1):
            spec = self.sub_study_spec(sub_study_name)
            pipeline.connect_input(
                spec.inverse_map('motion_mats'), merge_motion_mats,
                'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('tr'), merge_tr,
                'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('start_time'), merge_start_time,
                'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('real_duration'), merge_real_duration,
                'in{}'.format(i))

        md = pipeline.create_node(MeanDisplacementCalculation(),
                                  name='scan_time_info')
        md.inputs.input_names = input_names
        pipeline.connect(merge_motion_mats, 'out', md, 'motion_mats')
        pipeline.connect(merge_tr, 'out', md, 'trs')
        pipeline.connect(merge_start_time, 'out', md, 'start_times')
        pipeline.connect(merge_real_duration, 'out', md, 'real_durations')
        pipeline.connect_input('ref_brain', md, 'reference')
        pipeline.connect_output('mean_displacement', md, 'mean_displacement')
        pipeline.connect_output(
            'mean_displacement_rc', md, 'mean_displacement_rc')
        pipeline.connect_output(
            'mean_displacement_consecutive', md,
            'mean_displacement_consecutive')
        pipeline.connect_output('start_times', md, 'start_times')
        pipeline.connect_output('motion_par_rc', md, 'motion_parameters_rc')
        pipeline.connect_output('motion_par', md, 'motion_parameters')
        pipeline.connect_output('offset_indexes', md, 'offset_indexes')
        pipeline.connect_output('mats4average', md, 'mats4average')
        pipeline.connect_output('severe_motion_detection_report', md,
                                'corrupted_volumes')
        return pipeline

#     def motion_framing_pipeline(self, **kwargs):
#         return self.motion_framing_pipeline_factory(
#             pet_data_dir=None, pet_start_time=None, pet_duration=None,
#             **kwargs)

    def motion_framing_pipeline(self, **kwargs):

        inputs = [DatasetSpec('mean_displacement', text_format),
                  DatasetSpec('mean_displacement_consecutive', text_format),
                  DatasetSpec('start_times', text_format)]
        if 'pet_data_dir' in self.input_names:
            inputs.append(FieldSpec('pet_start_time', str))
            inputs.append(FieldSpec('pet_end_time', str))
        pipeline = self.create_pipeline(
            name='motion_framing',
            inputs=inputs,
            outputs=[DatasetSpec('frame_start_times', text_format),
                     DatasetSpec('frame_vol_numbers', text_format),
                     DatasetSpec('timestamps', directory_format)],
            desc=("Calculate when the head movement exceeded a "
                  "predefined threshold (default 2mm)."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        framing = pipeline.create_node(MotionFraming(), name='motion_framing')
        framing.inputs.motion_threshold = pipeline.option('framing_th')
        framing.inputs.temporal_threshold = pipeline.option(
            'framing_temporal_th')
        pipeline.connect_input('mean_displacement', framing,
                               'mean_displacement')
        pipeline.connect_input('mean_displacement_consecutive', framing,
                               'mean_displacement_consec')
        pipeline.connect_input('start_times', framing, 'start_times')
        if 'pet_data_dir' in self.input_names:
            pipeline.connect_input('pet_start_time', framing, 'pet_start_time')
            pipeline.connect_input('pet_end_time', framing, 'pet_end_time')
        pipeline.connect_output('frame_start_times', framing,
                                'frame_start_times')
        pipeline.connect_output('frame_vol_numbers', framing,
                                'frame_vol_numbers')
        pipeline.connect_output('timestamps', framing, 'timestamps_dir')
        return pipeline

    def plot_mean_displacement_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='plot_mean_displacement',
            inputs=[DatasetSpec('mean_displacement_rc', text_format),
                    DatasetSpec('offset_indexes', text_format),
                    DatasetSpec('frame_start_times', text_format)],
            outputs=[DatasetSpec('mean_displacement_plot', png_format)],
            desc=("Plot the mean displacement real clock"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        plot_md = pipeline.create_node(PlotMeanDisplacementRC(),
                                       name='plot_md')
        plot_md.inputs.framing = pipeline.option('md_framing')
        pipeline.connect_input('mean_displacement_rc', plot_md,
                               'mean_disp_rc')
        pipeline.connect_input('offset_indexes', plot_md,
                               'false_indexes')
        pipeline.connect_input('frame_start_times', plot_md,
                               'frame_start_times')
        pipeline.connect_output('mean_displacement_plot', plot_md,
                                'mean_disp_plot')
        return pipeline

    def frame_mean_transformation_mats_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='frame_mean_transformation_mats',
            inputs=[DatasetSpec('mats4average', text_format),
                    DatasetSpec('frame_vol_numbers', text_format)],
            outputs=[DatasetSpec('average_mats', directory_format)],
            desc=("Average all the transformation mats within each "
                  "detected frame."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        average = pipeline.create_node(AffineMatAveraging(),
                                       name='mats_averaging')
        pipeline.connect_input('frame_vol_numbers', average,
                               'frame_vol_numbers')
        pipeline.connect_input('mats4average', average,
                               'all_mats4average')
        pipeline.connect_output('average_mats', average,
                                'average_mats')
        return pipeline

    def pet_correction_factors_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='pet_correction_factors',
            inputs=[DatasetSpec('timestamps', directory_format)],
            outputs=[DatasetSpec('correction_factors', text_format)],
            desc=("Pipeline to calculate the correction factors to "
                  "account for frame duration when averaging the PET "
                  "frames to create the static PET image"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        corr_factors = pipeline.create_node(PetCorrectionFactor(),
                                            name='pet_corr_factors')
        pipeline.connect_input('timestamps', corr_factors,
                               'timestamps')
        pipeline.connect_output('correction_factors', corr_factors,
                                'corr_factors')
        return pipeline

    def static_frame2ref_alignment_pipeline(self, **kwargs):

        inputs = [DatasetSpec('average_mats', directory_format),
                  DatasetSpec('umap_ref_coreg_matrix', text_matrix_format),
                  DatasetSpec('umap_ref_qform_mat', text_matrix_format)]
        outputs = [DatasetSpec('static_frame2reference_mats',
                               directory_format)]
        umap = False
        if ('umap_ref' in self.sub_study_names and
                'umap' in self.input_names):
            inputs.append(DatasetSpec('umap', nifti_gz_format))
            outputs.append(DatasetSpec('umaps_align2ref', directory_format))
            umap = True

        pipeline = self.create_pipeline(
            name='static_frame2ref_alignment',
            inputs=inputs,
            outputs=outputs,
            desc=("Pipeline to create an affine mat to align each "
                  "detected frame to the reference. If umap is provided"
                  ", it will be also aligned to match the head position"
                  " in each frame and improve the static PET image "
                  "quality."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        frame_align = pipeline.create_node(
            FrameAlign2Reference(), name='static_frame2ref_alignment',
            requirements=[fsl509_req])
        frame_align.inputs.pct = pipeline.option('align_pct')
        frame_align.inputs.fixed_binning = False
        pipeline.connect_input('umap_ref_coreg_matrix', frame_align,
                               'ute_regmat')
        pipeline.connect_input('umap_ref_qform_mat', frame_align,
                               'ute_qform_mat')
        if not umap:
            pipeline.connect_input('average_mats', frame_align, 'average_mats')
            pipeline.connect_output('staitc_frame2reference_mats', frame_align,
                                    'frame2reference_mats')
        else:
            pipeline.connect_input('average_mats', frame_align, 'average_mats')
            pipeline.connect_output('staitc_frame2reference_mats', frame_align,
                                    'frame2reference_mats')
            pipeline.connect_input('umap', frame_align, 'umap')
            pipeline.connect_output('umaps_align2ref', frame_align,
                                    'umaps_align2ref')

        return pipeline

    def dynamic_frame2ref_alignment_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='dynamic_frame2ref_alignment',
            inputs=[DatasetSpec('fixed_binning_mats', directory_format),
                    DatasetSpec('umap_ref_coreg_matrix', text_matrix_format),
                    DatasetSpec('umap_ref_qform_mat', text_matrix_format)],
            outputs=[DatasetSpec('dynamic_frame2reference_mats',
                                 directory_format)],
            desc=("Pipeline to create an affine mat to align each "
                  "reconstructed bin to the reference."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        frame_align = pipeline.create_node(
            FrameAlign2Reference(), name='dynamic_frame2ref_alignment',
            requirements=[fsl509_req])
        frame_align.inputs.fixed_binning = True
        pipeline.connect_input('umap_ref_coreg_matrix', frame_align,
                               'ute_regmat')
        pipeline.connect_input('umap_ref_qform_mat', frame_align,
                               'ute_qform_mat')
        pipeline.connect_input('fixed_binning_mats', frame_align,
                               'average_mats')
        pipeline.connect_output('dynamic_frame2reference_mats',
                                frame_align, 'frame2reference_mats')

        return pipeline

    def nifti2dcm_conversion_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='conversion_to_dicom',
            inputs=[DatasetSpec('umaps_align2ref', directory_format),
                    DatasetSpec('umap', dicom_format)],
            outputs=[DatasetSpec('umap_aligned_dicoms', directory_format)],
            desc=(
                "Conversing aligned umap from nifti to dicom format - "
                "parallel implementation"),
            version=1,
            citations=(),
            **kwargs)

        list_niftis = pipeline.create_node(ListDir(), name='list_niftis')
        nii2dicom = pipeline.create_map_node(
            Nii2Dicom(), name='nii2dicom',
            iterfield=['in_file'], wall_time=20)
#         nii2dicom.inputs.extension = 'Frame'
        list_dicoms = pipeline.create_node(ListDir(), name='list_dicoms')
        list_dicoms.inputs.sort_key = dicom_fname_sort_key
        copy2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        copy2dir.inputs.extension = 'Frame'
        # Connect nodes
        pipeline.connect(list_niftis, 'files', nii2dicom, 'in_file')
        pipeline.connect(list_dicoms, 'files', nii2dicom, 'reference_dicom')
        pipeline.connect(nii2dicom, 'out_file', copy2dir, 'in_files')
        # Connect inputs
        pipeline.connect_input('umaps_align2ref', list_niftis, 'directory')
        pipeline.connect_input('umap', list_dicoms, 'directory')
        # Connect outputs
        pipeline.connect_output('umap_aligned_dicoms', copy2dir, 'out_dir')

        return pipeline

    def create_moco_series_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='create_moco_series',
            inputs=[DatasetSpec('start_times', text_format),
                    DatasetSpec('motion_par', text_format)],
            outputs=[DatasetSpec('moco_series', directory_format)],
            desc=("Pipeline to generate a moco_series that can be then "
                  "imported back in the scanner and used to correct the"
                  " pet data"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        moco = pipeline.create_node(CreateMocoSeries(),
                                    name='create_moco_series')
        pipeline.connect_input('start_times', moco, 'start_times')
        pipeline.connect_input('motion_par', moco, 'motion_par')
        moco.inputs.moco_template = pipeline.option('moco_template')

        pipeline.connect_output('moco_series', moco, 'modified_moco')
        return pipeline

    def gather_outputs_pipeline(self, **kwargs):
        inputs = [DatasetSpec('mean_displacement_plot', png_format),
                  DatasetSpec('motion_par', text_format),
                  DatasetSpec('correction_factors', text_format),
                  DatasetSpec('severe_motion_detection_report', text_format),
                  DatasetSpec('timestamps', directory_format)]
        if ('umap_ref' in self.sub_study_names and
                'umap' in self.input_names):
            inputs.append(
                DatasetSpec('frame2reference_mats', directory_format))
            inputs.append(DatasetSpec('umap_ref_preproc', nifti_gz_format))
            inputs.append(
                DatasetSpec('umap_aligned_dicoms', directory_format))
        if ('umap_ref' in self.sub_study_names and
                'umap' not in self.input_names):
            inputs.append(
                DatasetSpec('frame2reference_mats', directory_format))
            inputs.append(DatasetSpec('umap_ref_preproc', nifti_gz_format))

        pipeline = self.create_pipeline(
            name='gather_motion_detection_outputs',
            inputs=inputs,
            outputs=[DatasetSpec('motion_detection_output', directory_format)],
            desc=("Pipeline to gather together all the outputs from "
                  "the motion detection pipeline."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        merge_inputs = pipeline.create_node(Merge(len(inputs)),
                                            name='merge_inputs')
        for i, dataset in enumerate(inputs, start=1):
            pipeline.connect_input(
                dataset.name, merge_inputs, 'in{}'.format(i))

        copy2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        pipeline.connect(merge_inputs, 'out', copy2dir, 'in_files')

        pipeline.connect_output('motion_detection_output', copy2dir, 'out_dir')
        return pipeline

    prepare_pet_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_data_preparation_pipeline')

    pet_header_info_extraction_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_time_info_extraction_pipeline')

    def static_motion_correction_pipeline(self, **kwargs):
        inputs = [DatasetSpec('pet_data_prepared', directory_format),
                  DatasetSpec('static_frame2reference_mats', directory_format),
                  DatasetSpec('correction_factors', text_format),
                  DatasetSpec('umap_ref_preproc', nifti_gz_format)]
        if 'Struct2Align' in self.input_names:
            inputs.append(DatasetSpec('Struct2Align', nifti_gz_format))
            StructAlignment = True

        pipeline = self.create_pipeline(
            name='static_mc',
            inputs=inputs,
            outputs=[DatasetSpec('static_motion_correction_results',
                                 directory_format)],
            desc=("Given a folder with reconstructed PET data, this "
                  "pipeline will generate a motion corrected static PET"
                  "image using information extracted from the MR-based "
                  "motion detection pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        check_pet = pipeline.create_node(CheckPetMCInputs(),
                                         name='check_pet_data')
        pipeline.connect_input('pet_data_prepared', check_pet, 'pet_data')
        pipeline.connect_input('static_frame2reference_mats', check_pet,
                               'motion_mats')
        pipeline.connect_input('correction_factors', check_pet, 'corr_factors')
        pet_mc = pipeline.create_map_node(
            PetImageMotionCorrection(), name='pet_mc',
            iterfield=['pet_image', 'motion_mat', 'corr_factor'])
        pipeline.connect(check_pet, 'pet_images', pet_mc, 'pet_image')
        pipeline.connect(check_pet, 'motion_mats', pet_mc, 'motion_mat')
        pipeline.connect(check_pet, 'corr_factors', pet_mc, 'corr_factor')
        pipeline.connect_input('umap_ref_preproc', pet_mc, 'ute_image')
        if StructAlignment:
            struct_reg = pipeline.create_node(FLIRT(),
                                              name='ute2structural_reg')
            pipeline.connect_input('umap_ref_preproc', struct_reg, 'in_file')
            pipeline.connect_input('Struct2Align', struct_reg, 'reference')
            struct_reg.inputs.dof = 6
            struct_reg.inputs.cost_func = 'normmi'
            struct_reg.inputs.cost = 'normmi'
            pipeline.connect(struct_reg, 'out_matrix_file', pet_mc,
                             'ute2structural_regmat')
            struct_qform = pipeline.create_node(FLIRT(),
                                                name='ute2structural_reg')
            pipeline.connect_input('umap_ref_preproc', struct_qform, 'in_file')
            pipeline.connect_input('Struct2Align', struct_qform, 'reference')
            struct_qform.inputs.uses_qform = True
            struct_qform.inputs.apply_xfm = True
            pipeline.connect(struct_qform, 'out_matrix_file', pet_mc,
                             'ute2structural_qform')
            pipeline.connect_input('Struct2Align', pet_mc,
                                   'structural_image')
        static_im_gen = pipeline.create_node(StaticPETImageGeneration(),
                                             name='static_mc_generation')
        pipeline.connect(pet_mc, 'pet_mc_images', static_im_gen,
                         'pet_mc_images')
        pipeline.connect(pet_mc, 'pet_mc_ps_images', static_im_gen,
                         'pet_mc_ps_images')
        pipeline.connect(pet_mc, 'pet_no_mc_images', static_im_gen,
                         'pet_no_mc_images')
        cropping = pipeline.create_node(PETFovCropping(), name='pet_cropping')
        cropping.inputs.x_min = pipeline.option('crop_xmin')
        cropping.inputs.x_size = pipeline.option('crop_xsize')
        cropping.inputs.y_min = pipeline.option('crop_ymin')
        cropping.inputs.y_size = pipeline.option('crop_ysize')
        cropping.inputs.z_min = pipeline.option('crop_zmin')
        cropping.inputs.z_size = pipeline.option('crop_zsize')
        pipeline.connect(static_im_gen, 'static_mc_ps', cropping, 'pet_image')

        merge_outputs = pipeline.create_node(Merge(3), name='merge_outputs')
        pipeline.connect(static_im_gen, 'static_mc', merge_outputs, 'in1')
        pipeline.connect(static_im_gen, 'static_no_mc', merge_outputs, 'in2')
        pipeline.connect(cropping, 'pet_cropped', merge_outputs, 'in3')

        copy2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        pipeline.connect(merge_outputs, 'out', copy2dir, 'in_files')

        pipeline.connect_output('static_motion_correction_results', copy2dir,
                                'out_dir')

        return pipeline

    def dynamic_motion_correction_pipeline(self, **kwargs):
        inputs = [DatasetSpec('pet_data_prepared', directory_format),
                  DatasetSpec('dynamic_frame2reference_mats', directory_format),
                  DatasetSpec('umap_ref_preproc', nifti_gz_format)]
        if 'Struct2Align' in self.input_names:
            inputs.append(DatasetSpec('Struct2Align', nifti_gz_format))
            StructAlignment = True

        pipeline = self.create_pipeline(
            name='dynamic_mc',
            inputs=inputs,
            outputs=[DatasetSpec('dynamic_motion_correction_results',
                                 directory_format)],
            desc=("Given a folder with reconstructed PET data, this "
                  "pipeline will generate a motion corrected static PET"
                  "image using information extracted from the MR-based "
                  "motion detection pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)
        check_pet = pipeline.create_node(CheckPetMCInputs(),
                                         name='check_pet_data')
        pipeline.connect_input('pet_data_prepared', check_pet, 'pet_data')
        pipeline.connect_input('dynamic_frame2reference_mats', check_pet,
                               'motion_mats')
        pet_mc = pipeline.create_map_node(
            PetImageMotionCorrection(), name='pet_mc',
            iterfield=['pet_image', 'motion_mat'])
        pipeline.connect(check_pet, 'pet_images', pet_mc, 'pet_image')
        pipeline.connect(check_pet, 'motion_mats', pet_mc, 'motion_mat')
        pipeline.connect_input('umap_ref_preproc', pet_mc, 'ute_image')
        if StructAlignment:
            struct_reg = pipeline.create_node(FLIRT(),
                                              name='ute2structural_reg')
            pipeline.connect_input('umap_ref_preproc', struct_reg, 'in_file')
            pipeline.connect_input('Struct2Align', struct_reg, 'reference')
            struct_reg.inputs.dof = 6
            struct_reg.inputs.cost_func = 'normmi'
            struct_reg.inputs.cost = 'normmi'
            pipeline.connect(struct_reg, 'out_matrix_file', pet_mc,
                             'ute2structural_regmat')
            struct_qform = pipeline.create_node(FLIRT(),
                                                name='ute2structural_reg')
            pipeline.connect_input('umap_ref_preproc', struct_qform, 'in_file')
            pipeline.connect_input('Struct2Align', struct_qform, 'reference')
            struct_qform.inputs.uses_qform = True
            struct_qform.inputs.apply_xfm = True
            pipeline.connect(struct_qform, 'out_matrix_file', pet_mc,
                             'ute2structural_qform')
            pipeline.connect_input('Struct2Align', pet_mc,
                                   'structural_image')
        merge_mc = pipeline.create_node(Merge(), name='merge_pet_mc')
        pipeline.connect(pet_mc, 'pet_mc_images', merge_mc, 'in_files')
        merge_mc.inputs.dimension = 't'
        merge_no_mc = pipeline.create_node(Merge(), name='merge_pet_no_mc')
        pipeline.connect(pet_mc, 'pet_no_mc_images', merge_no_mc, 'in_files')
        merge_no_mc.inputs.dimension = 't'
        cropping = pipeline.create_map_node(
            PETFovCropping(), name='pet_cropping', iterfield=['pet_image'])
        cropping.inputs.x_min = pipeline.option('crop_xmin')
        cropping.inputs.x_size = pipeline.option('crop_xsize')
        cropping.inputs.y_min = pipeline.option('crop_ymin')
        cropping.inputs.y_size = pipeline.option('crop_ysize')
        cropping.inputs.z_min = pipeline.option('crop_zmin')
        cropping.inputs.z_size = pipeline.option('crop_zsize')
        pipeline.connect(pet_mc, 'pet_mc_ps_images', cropping, 'pet_image')
        merge_mc_ps = pipeline.create_node(Merge(), name='merge_pet_mc_ps')
        pipeline.connect(cropping, 'pet_cropped', merge_mc_ps, 'in_files')
        merge_mc_ps.inputs.dimension = 't'
        merge_outputs = pipeline.create_node(Merge(3), name='merge_outputs')
        pipeline.connect(merge_mc, 'merged_file', merge_outputs, 'in1')
        pipeline.connect(merge_no_mc, 'merged_file', merge_outputs, 'in2')
        pipeline.connect(merge_mc_ps, 'merged_file', merge_outputs, 'in3')

        copy2dir = pipeline.create_node(CopyToDir(), name='copy2dir')
        pipeline.connect(merge_outputs, 'out', copy2dir, 'in_files')

        pipeline.connect_output('dynamic_motion_correction_results', copy2dir,
                                'out_dir')

        return pipeline

    def fixed_binning_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='fixed_binning',
            inputs=[DatasetSpec('start_times', text_format),
                    FieldSpec('pet_start_time', str),
                    FieldSpec('pet_duration', int),
                    DatasetSpec('mats4average', text_format)],
            outputs=[DatasetSpec('fixed_binning_mats', directory_format)],
            desc=("Pipeline to generate average motion matrices for "
                  "each bin in a dynamic PET reconstruction experiment."
                  "This will be the input for the dynamic motion correction."),
            version=1,
            citations=[fsl_cite], **kwargs)

        binning = pipeline.create_node(FixedBinning(), name='fixed_binning')
        pipeline.connect_input('start_times', binning, 'start_times')
        pipeline.connect_input('pet_start_time', binning, 'pet_start_time')
        pipeline.connect_input('pet_duration', binning, 'pet_duration')
        pipeline.connect_input('mats4average', binning, 'motion_mats')
        binning.inputs.n_frames = pipeline.option('fixed_binning_n_frames')
        binning.inputs.pet_offset = pipeline.option('fixed_binning_pet_offset')
        binning.inputs.bin_len = pipeline.option('fixed_binning_bin_len')

        pipeline.connect_output('fixed_binning_mats', binning,
                                'average_bin_mats')

        pipeline.assert_connected()
        return pipeline


def create_motion_correction_class(name, ref=None, ref_type=None, t1s=None,
                                   t2s=None, dmris=None, epis=None,
                                   umaps=None, dynamic=False, umap_ref=None,
                                   pet_data_dir=None, pet_recon_dir=None):

    inputs = []
    dct = {}
    data_specs = []
    run_pipeline = False
    option_specs = [OptionSpec('ref_preproc_resolution', [1])]

    if pet_data_dir is not None:
        inputs.append(DatasetMatch('pet_data_dir', directory_format,
                                   pet_data_dir))
    if pet_recon_dir is not None:
        inputs.append(DatasetMatch('pet_data_reconstructed', directory_format,
                                   pet_recon_dir))
    if pet_data_dir is not None and pet_recon_dir is not None and dynamic:
        output_data = 'dynamic_motion_correction_results'
    elif pet_data_dir is not None and pet_recon_dir is not None and not dynamic:
        output_data = 'static_motion_correction_results'
    else:
        output_data = 'motion_detection_output'
            
    if not ref:
        raise Exception('A reference image must be provided!')
    if ref_type == 't1':
        ref_study = MotionReferenceT1Study
    elif ref_type == 't2':
        ref_study = MotionReferenceT2Study
    else:
        raise Exception('{} is not a recognized ref_type!The available '
                        'ref_types are t1 or t2.'.format(ref_type))

    study_specs = [SubStudySpec('ref', ref_study)]
    ref_spec = {'ref_brain': 'coreg_ref_brain'}
    inputs.append(DatasetMatch('ref_primary', dicom_format, ref))

    if not umap_ref:
        logger.info(
            'Umap reference not provided. The matrices that realign the PET'
            ' image in each detected frame to the reference cannot be '
            'generated. See documentation for further information.')
    else:
        if umap_ref in t1s:
            umap_ref_study = T1Study
            t1s.remove(umap_ref)
        elif umap_ref in t2s:
            umap_ref_study = T2Study
            t2s.remove(umap_ref)
        else:
            umap_ref = None

    if t1s:
        study_specs.extend(
                [SubStudySpec('t1_{}'.format(i), T1Study,
                              ref_spec) for i in range(len(t1s))])
        inputs.extend(
            DatasetMatch('t1_{}_primary'.format(i), dicom_format, t1_scan)
            for i, t1_scan in enumerate(t1s))
        run_pipeline = True

    if t2s:
        study_specs.extend(
                [SubStudySpec('t2_{}'.format(i), T2Study,
                              ref_spec) for i in range(len(t2s))])
        inputs.extend(DatasetMatch('t2_{}_primary'.format(i), dicom_format,
                                   t2_scan)
                      for i, t2_scan in enumerate(t2s))
        run_pipeline = True

    if umap_ref and not umaps:
        logger.info('Umap not provided. The umap realignment will not be '
                    'performed. Matrices that realign each detected frame to '
                    'the reference will be calculated.')
        study_specs.append(SubStudySpec('umap_ref', umap_ref_study, ref_spec))
        inputs.append(DatasetMatch('umap_ref_primary', dicom_format, umap_ref))

    elif umap_ref and umaps:
        logger.info('Umap will be realigned to match the head position in '
                    'each frame. Matrices that realign each frame to the '
                    'reference will be calculated.')
        if len(umaps) > 1:
            logger.info('More than one umap provided. Only the first one will '
                        'be used.')
        umaps = umaps[0]
        study_specs.append(SubStudySpec('umap_ref', umap_ref_study, ref_spec))
        inputs.append(DatasetMatch('umap_ref_primary', dicom_format, umap_ref))
        inputs.append(DatasetMatch('umap', dicom_format, umaps))

        run_pipeline = True

    elif not umap_ref and umaps:
        logger.warning('Umap provided without corresponding reference image. '
                       'Realignment cannot be performed without UTE. Umap will'
                       'be ignored.')

    if epis:
        epi_refspec = ref_spec.copy()
        epi_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                            'ref_preproc': 'coreg_ref_preproc'})
        study_specs.extend(SubStudySpec('epi_{}'.format(i), EPIStudy,
                                        epi_refspec)
                           for i in range(len(epis)))
        inputs.extend(
            DatasetMatch('epi_{}_primary'.format(i), dicom_format, epi_scan)
            for i, epi_scan in enumerate(epis))
        run_pipeline = True
    if dmris:
        unused_dwi = []
        dmris_main = [x for x in dmris if x[-1] == '0']
        dmris_ref = [x for x in dmris if x[-1] == '1']
        dmris_opposite = [x for x in dmris if x[-1] == '-1']
        b0_refspec = ref_spec.copy()
        b0_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                           'ref_preproc': 'coreg_ref_preproc'})
        if dmris_main and not dmris_opposite:
            logger.warning(
                'No opposite phase encoding direction b0 provided. DWI '
                'motion correction will be performed without distortion '
                'correction. THIS IS SUB-OPTIMAL!')
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DWIStudy, ref_spec)
                for i in range(len(dmris_main)))
            inputs.extend(
                DatasetMatch('dwi_{}_primary'.format(i), dicom_format,
                             dmris_main_scan[0])
                for i, dmris_main_scan in enumerate(dmris_main))
        if dmris_main and dmris_opposite:
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DWIStudy, ref_spec)
                for i in range(len(dmris_main)))
            inputs.extend(
                DatasetMatch('dwi_{}_primary'.format(i), dicom_format,
                             dmris_main[i][0]) for i in range(len(dmris_main)))
            if len(dmris_main) <= len(dmris_opposite):
                inputs.extend(DatasetMatch('dwi_{}_dwi_reference'.format(i),
                                           dicom_format, dmris_opposite[i][0])
                              for i in range(len(dmris_main)))
            else:
                inputs.extend(DatasetMatch('dwi_{}_dwi_reference'.format(i),
                                           dicom_format, dmris_opposite[0][0])
                              for i in range(len(dmris_main)))
        if dmris_opposite and dmris_main and not dmris_ref:
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EPIStudy, b0_refspec)
                for i in range(len(dmris_opposite)))
            inputs.extend(DatasetMatch('b0_{}_primary'.format(i),
                                       dicom_format, dmris_opposite[i][0])
                          for i in range(len(dmris_opposite)))
            if len(dmris_opposite) <= len(dmris_main):
                inputs.extend(DatasetMatch('b0_{}_reverse_phase'.format(i),
                                           dicom_format, dmris_main[i][0])
                              for i in range(len(dmris_opposite)))
            else:
                inputs.extend(DatasetMatch('b0_{}_reverse_phase'.format(i),
                                           dicom_format, dmris_main[0][0])
                              for i in range(len(dmris_opposite)))
        elif dmris_opposite and dmris_ref:
            min_index = min(len(dmris_opposite), len(dmris_ref))
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EPIStudy, b0_refspec)
                for i in range(min_index*2))
            inputs.extend(
                DatasetMatch('b0_{}_primary'.format(i), dicom_format,
                             scan[0])
                for i, scan in enumerate(dmris_opposite[:min_index] +
                                         dmris_ref[:min_index]))
            inputs.extend(
                DatasetMatch('b0_{}_reverse_phase'.format(i), dicom_format,
                             scan[0])
                for i, scan in enumerate(dmris_ref[:min_index] +
                                         dmris_opposite[:min_index]))
            unused_dwi = [scan for scan in dmris_ref[min_index:] +
                          dmris_opposite[min_index:]]
        elif dmris_opposite or dmris_ref:
            unused_dwi = [scan for scan in dmris_ref + dmris_opposite]
        if unused_dwi:
            logger.info(
                'The following scans:\n{}\nwere not assigned during the DWI '
                'motion detection initialization (probably a different number '
                'of main DWI scans and b0 images was provided). They will be '
                'processed os "other" scans.'
                .format('\n'.join(s[0] for s in unused_dwi)))
            study_specs.extend(
                SubStudySpec('t2_{}'.format(i), T2Study, ref_spec)
                for i in range(len(t2s), len(t2s)+len(unused_dwi)))
            inputs.extend(
                DatasetMatch('t2_{}_primary'.format(i), dicom_format, scan[0])
                for i, scan in enumerate(unused_dwi, start=len(t2s)))
        run_pipeline = True

    if not run_pipeline:
        raise Exception('At least one scan, other than the reference, must be '
                        'provided!')

    dct['add_sub_study_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['__metaclass__'] = MultiStudyMetaClass
    dct['add_option_specs'] = option_specs
    return (MultiStudyMetaClass(name, (MotionDetectionMixin,), dct), inputs,
            output_data)
