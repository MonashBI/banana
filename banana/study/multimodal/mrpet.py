from arcana.data import FilesetSpec, FieldSpec
from banana.file_format import (
    nifti_gz_format, directory_format, text_format, png_format, dicom_format,
    text_matrix_format)
from banana.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming, PlotMeanDisplacementRC,
    AffineMatAveraging, PetCorrectionFactor, CreateMocoSeries, FixedBinning,
    UmapAlign2Reference, ReorientUmap)
from banana.citation import fsl_cite
from arcana.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from banana.study.mri.epi import EpiStudy
from banana.study.mri.t1 import T1Study
from banana.study.mri.t2 import T2Study
from nipype.interfaces.utility import Merge
from banana.study.mri.dwi import DwiStudy
from banana.requirement import fsl_req, mrtrix_req, ants_req
from arcana.exceptions import ArcanaNameError
from arcana.data import FilesetSelector
import logging
from banana.study.pet.base import PetStudy
from banana.interfaces.custom.pet import (
    CheckPetMCInputs, PetImageMotionCorrection, StaticPETImageGeneration,
    PETFovCropping)
from arcana.study import ParameterSpec, SwitchSpec
import os
from banana.interfaces.converters import Nii2Dicom
from arcana.utils.interfaces import CopyToDir, ListDir, dicom_fname_sort_key
from nipype.interfaces.fsl.preprocess import FLIRT
import nipype.interfaces.fsl as fsl
from nipype.interfaces.fsl.utils import ImageMaths
from banana.interfaces.ants import AntsRegSyn
from nipype.interfaces.ants.resampling import ApplyTransforms


logger = logging.getLogger('Arcana')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)

reference_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../',
                 'reference_data'))

template_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__).split('banana')[0],
                 'banana', 'banana', 'templates'))


class MotionDetectionMixin(MultiStudy, metaclass=MultiStudyMetaClass):

    add_sub_study_specs = [
        SubStudySpec('pet_mc', PetStudy, {
            'pet_data_dir': 'pet_data_dir',
            'pet_data_reconstructed': 'pet_recon_dir',
            'pet_data_prepared': 'pet_recon_dir_prepared',
            'pet_start_time': 'pet_start_time',
            'pet_end_time': 'pet_end_time',
            'pet_duration': 'pet_duration'})]

    add_data_specs = [
        FilesetSpec('pet_data_dir', directory_format, optional=True),
        FilesetSpec('pet_data_reconstructed', directory_format, optional=True),
        FilesetSpec('struct2align', nifti_gz_format, optional=True),
        FilesetSpec('pet_data_prepared', directory_format,
                    'prepare_pet_pipeline'),
        FilesetSpec('static_motion_correction_results', directory_format,
                    'motion_correction_pipeline'),
        FilesetSpec('dynamic_motion_correction_results', directory_format,
                    'motion_correction_pipeline'),
        FilesetSpec('umap', dicom_format, optional=True),
        FilesetSpec('mean_displacement', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('mean_displacement_rc', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('mean_displacement_consecutive', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('mats4average', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('start_times', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('motion_par_rc', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('motion_par', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('offset_indexes', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('severe_motion_detection_report', text_format,
                    'mean_displacement_pipeline'),
        FilesetSpec('frame_start_times', text_format,
                    'motion_framing_pipeline'),
        FilesetSpec('frame_vol_numbers', text_format,
                    'motion_framing_pipeline'),
        FilesetSpec('timestamps', directory_format,
                    'motion_framing_pipeline'),
        FilesetSpec('mean_displacement_plot', png_format,
                    'plot_mean_displacement_pipeline'),
        FilesetSpec('rotation_plot', png_format,
                    'plot_mean_displacement_pipeline'),
        FilesetSpec('translation_plot', png_format,
                    'plot_mean_displacement_pipeline'),
        FilesetSpec('average_mats', directory_format,
                    'frame_mean_transformation_mats_pipeline'),
        FilesetSpec('correction_factors', text_format,
                    'pet_correction_factors_pipeline'),
        FilesetSpec('umaps_align2ref', directory_format,
                    'umap_realignment_pipeline'),
        FilesetSpec('umap_aligned_dicoms', directory_format,
                    'nifti2dcm_conversion_pipeline'),
        FilesetSpec('motion_detection_output', directory_format,
                    'gather_outputs_pipeline'),
        FilesetSpec('moco_series', directory_format,
                    'create_moco_series_pipeline'),
        FilesetSpec('fixed_binning_mats', directory_format,
                    'fixed_binning_pipeline'),
        FieldSpec('pet_duration', dtype=int,
                  pipeline_name='pet_header_extraction_pipeline'),
        FieldSpec('pet_end_time', dtype=str,
                  pipeline_name='pet_header_extraction_pipeline'),
        FieldSpec('pet_start_time', dtype=str,
                  pipeline_name='pet_header_extraction_pipeline')]

    add_parameter_specs = [
        ParameterSpec('framing_th', 2.0),
        ParameterSpec('framing_temporal_th', 30.0),
        ParameterSpec('framing_duration', 0),
        ParameterSpec('md_framing', True),
        ParameterSpec('align_pct', False),
        ParameterSpec('align_fixed_binning', False),
        ParameterSpec('moco_template', os.path.join(
            reference_path, 'moco_template.IMA')),
        ParameterSpec('PET_template_MNI', os.path.join(
            template_path, 'PET_template_MNI.nii.gz')),
        ParameterSpec('fixed_binning_n_frames', 0),
        ParameterSpec('pet_offset', 0),
        ParameterSpec('fixed_binning_bin_len', 60),
        ParameterSpec('crop_xmin', 100),
        ParameterSpec('crop_xsize', 130),
        ParameterSpec('crop_ymin', 100),
        ParameterSpec('crop_ysize', 130),
        ParameterSpec('crop_zmin', 20),
        ParameterSpec('crop_zsize', 100),
        ParameterSpec('PET2MNI_reg', False),
        ParameterSpec('dynamic_pet_mc', False)]

    def mean_displacement_pipeline(self, **kwargs):
        inputs = [FilesetSpec('ref_brain', nifti_gz_format)]
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
                        'magnitude')).pattern)
                sub_study_names.append(sub_study_spec.name)
            except ArcanaNameError:
                continue  # Sub study doesn't have motion mat

#             inputs=inputs,
#             outputs=[FilesetSpec('mean_displacement', text_format),
#                      FilesetSpec('mean_displacement_rc', text_format),
#                      FilesetSpec('mean_displacement_consecutive', text_format),
#                      FilesetSpec('start_times', text_format),
#                      FilesetSpec('motion_par_rc', text_format),
#                      FilesetSpec('motion_par', text_format),
#                      FilesetSpec('offset_indexes', text_format),
#                      FilesetSpec('mats4average', text_format),
#                      FilesetSpec('severe_motion_detection_report',
#                                  text_format)],

        pipeline = self.new_pipeline(
            name='mean_displacement_calculation',
            desc=("Calculate the mean displacement between each motion"
                  " matrix and a reference."),
            citations=[fsl_cite],
            **kwargs)

        num_motion_mats = len(sub_study_names)

        merge_motion_mats = pipeline.add(
            'merge_motion_mats',
            Merge(num_motion_mats))

        merge_tr = pipeline.add(
            'merge_tr',
            Merge(num_motion_mats))

        merge_start_time = pipeline.add(
            'merge_start_time',
            Merge(num_motion_mats))

        merge_real_duration = pipeline.add(
            'merge_real_duration',
            Merge(num_motion_mats))

        for i, sub_study_name in enumerate(sub_study_names, start=1):
            spec = self.sub_study_spec(sub_study_name)
            pipeline.connect_input(
                spec.inverse_map('motion_mats'), merge_motion_mats,
                'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('tr'), merge_tr, 'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('start_time'), merge_start_time,
                'in{}'.format(i))
            pipeline.connect_input(
                spec.inverse_map('real_duration'), merge_real_duration,
                'in{}'.format(i))

        md = pipeline.add(
            'scan_time_info',
            MeanDisplacementCalculation(
                input_names=input_names),
            inputs={
                'motion_mats': (merge_motion_mats, 'out'),  # internal md
                'trs': (merge_tr, 'out'),  # internal md
                'start_times': (merge_start_time, 'out'),  # internal md
                'real_durations': (merge_real_duration, 'out'),  # internal md
                'reference': ('ref_brain', _format)},
            outputs={
                'mean_displacement': ('mean_displacement', _format),  # output md
                'mean_displacement_rc': ('mean_displacement_rc', _format),  # output md
                'mean_displacement_consecutive': ('mean_displacement_consecutive', _format),  # output md
                'start_times': ('start_times', _format),  # output md
                'motion_par_rc': ('motion_parameters_rc', _format),  # output md
                'motion_par': ('motion_parameters', _format),  # output md
                'offset_indexes': ('offset_indexes', _format),  # output md
                'mats4average': ('mats4average', _format),  # output md
                'severe_motion_detection_report': ('corrupted_volumes', _format)})
                ,  #  md parameter
                ,  # input md
                ,  # output md
        return pipeline

    def motion_framing_pipeline(self, **kwargs):

        inputs = [FilesetSpec('mean_displacement', text_format),
                  FilesetSpec('mean_displacement_consecutive', text_format),
                  FilesetSpec('start_times', text_format)]
        if 'pet_data_dir' in self.input_names:
            inputs.append(FieldSpec('pet_start_time', str))
            inputs.append(FieldSpec('pet_end_time', str))
#             inputs=inputs,
#             outputs=[FilesetSpec('frame_start_times', text_format),
#                      FilesetSpec('frame_vol_numbers', text_format),
#                      FilesetSpec('timestamps', directory_format)],

        pipeline = self.new_pipeline(
            name='motion_framing',
            desc=("Calculate when the head movement exceeded a "
                  "predefined threshold (default 2mm)."),
            citations=[fsl_cite],
            **kwargs)

        framing = pipeline.add(
            'motion_framing',
            MotionFraming(
                motion_threshold=self.parameter('framing_th'),  #  framing parameter
                temporal_threshold=self.parameter('framing_temporal_th'),  #  framing parameter
                pet_offset=self.parameter('pet_offset'),  #  framing parameter
                pet_duration=self.parameter('framing_duration')),
            inputs={
                'mean_displacement': ('mean_displacement', _format),  # input framing
                'mean_displacement_consec': ('mean_displacement_consecutive', _format),  # input framing
                'start_times': ('start_times', _format)},
            outputs={
                'frame_start_times': ('frame_start_times', _format),  # output framing
                'frame_vol_numbers': ('frame_vol_numbers', _format),  # output framing
                'timestamps': ('timestamps_dir', _format)})
                ,  #  framing parameter
                ,  # input framing
                ,  # output framing
        if 'pet_data_dir' in self.input_names:
            pipeline.connect_input('pet_start_time', framing,
                                   'pet_start_time')
            pipeline.connect_input('pet_end_time', framing,
                                   'pet_end_time')
        return pipeline

    def plot_mean_displacement_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('mean_displacement_rc', text_format),
#                     FilesetSpec('offset_indexes', text_format),
#                     FilesetSpec('frame_start_times', text_format)],
#             outputs=[FilesetSpec('mean_displacement_plot', png_format)],

        pipeline = self.new_pipeline(
            name='plot_mean_displacement',
            desc=("Plot the mean displacement real clock"),
            citations=[fsl_cite],
            **kwargs)

        plot_md = pipeline.add(
            'plot_md',
            PlotMeanDisplacementRC(
                framing=self.parameter('md_framing')),
            inputs={
                'mean_disp_rc': ('mean_displacement_rc', _format),  # input plot_md
                'false_indexes': ('offset_indexes', _format),  # input plot_md
                'frame_start_times': ('frame_start_times', _format),  # input plot_md
                'motion_par_rc': ('motion_par_rc', _format)},
            outputs={
                'mean_displacement_plot': ('mean_disp_plot', _format),  # output plot_md
                'rotation_plot': ('rot_plot', _format),  # output plot_md
                'translation_plot': ('trans_plot', _format)})
                ,  #  plot_md parameter
                ,  # input plot_md
                ,  # output plot_md
        return pipeline

    def frame_mean_transformation_mats_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('mats4average', text_format),
#                     FilesetSpec('frame_vol_numbers', text_format)],
#             outputs=[FilesetSpec('average_mats', directory_format)],

        pipeline = self.new_pipeline(
            name='frame_mean_transformation_mats',
            desc=("Average all the transformation mats within each "
                  "detected frame."),
            citations=[fsl_cite],
            **kwargs)

        average = pipeline.add(
            'mats_averaging',
            AffineMatAveraging(),
            inputs={
                'frame_vol_numbers': ('frame_vol_numbers', _format),  # input average
                'all_mats4average': ('mats4average', _format)},
            outputs={
                'average_mats': ('average_mats', _format)})
                ,  # input average
                ,  # output average
        return pipeline

    def fixed_binning_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('start_times', text_format),
#                     FieldSpec('pet_start_time', str),
#                     FieldSpec('pet_duration', int),
#                     FilesetSpec('mats4average', text_format)],
#             outputs=[FilesetSpec('fixed_binning_mats', directory_format)],

        pipeline = self.new_pipeline(
            name='fixed_binning',
            desc=("Pipeline to generate average motion matrices for "
                  "each bin in a dynamic PET reconstruction experiment."
                  "This will be the input for the dynamic motion correction."),
            citations=[fsl_cite], **kwargs)

        binning = pipeline.add(
            'fixed_binning',
            FixedBinning(
                n_frames=self.parameter('fixed_binning_n_frames'),  #  binning parameter
                pet_offset=self.parameter('pet_offset'),  #  binning parameter
                bin_len=self.parameter('fixed_binning_bin_len')),
            inputs={
                'start_times': ('start_times', _format),  # input binning
                'pet_start_time': ('pet_start_time', _format),  # input binning
                'pet_duration': ('pet_duration', _format),  # input binning
                'motion_mats': ('mats4average', _format)},
            outputs={
                'fixed_binning_mats': ('average_bin_mats', _format)})
                ,  # input binning
                ,  #  binning parameter
                ,  # output binning

        return pipeline

    def pet_correction_factors_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('timestamps', directory_format)],
#             outputs=[FilesetSpec('correction_factors', text_format)],

        pipeline = self.new_pipeline(
            name='pet_correction_factors',
            desc=("Pipeline to calculate the correction factors to "
                  "account for frame duration when averaging the PET "
                  "frames to create the static PET image"),
            citations=[fsl_cite],
            **kwargs)

        corr_factors = pipeline.add(
            'pet_corr_factors',
            PetCorrectionFactor(),
            inputs={
                'timestamps': ('timestamps', _format)},
            outputs={
                'correction_factors': ('corr_factors', _format)})
                ,  # input corr_factors
                ,  # output corr_factors
        return pipeline

    def nifti2dcm_conversion_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('umaps_align2ref', directory_format),
#                     FilesetSpec('umap', dicom_format)],
#             outputs=[FilesetSpec('umap_aligned_dicoms', directory_format)],

        pipeline = self.new_pipeline(
            name='conversion_to_dicom',
            desc=(
                "Conversing aligned umap from nifti to dicom format - "
                "parallel implementation"),
            citations=(),
            **kwargs)

        list_niftis = pipeline.add(
            'list_niftis',
            ListDir(),
            inputs={
                'directory': ('umaps_align2ref', _format)},
            outputs={
                })
                ,  # input list_niftis

        reorient_niftis = pipeline.add(
            'reorient_niftis',
            ReorientUmap(),
            inputs={
                'niftis': (list_niftis, 'files'),  # internal reorient_niftis
                'umap': ('umap', _format)},
            outputs={
                },
            requirements=[mrtrix_req.v('3.0rc3')])
                ,  # input reorient_niftis

        list_dicoms = pipeline.add(
            'list_dicoms',
            ListDir(
                sort_key=dicom_fname_sort_key),
            inputs={
                'directory': ('umap', _format)},
            outputs={
                })
                ,  #  list_dicoms parameter
                ,  # input list_dicoms

        nii2dicom = pipeline.add(
            'nii2dicom',
            Nii2Dicom(),
            inputs={
                'reference_dicom': (list_dicoms, 'files')},
            outputs={
                'in_file': (reorient_niftis, 'reoriented_umaps')},
            iterfield=['in_file'],
            wall_time=20)
#                 extension='Frame',  #  nii2dicom parameter
                ,  # internal nii2dicom
                ,  # internal nii2dicom

        copy2dir = pipeline.add(
            'copy2dir',
            CopyToDir(
                extension='Frame'),
            inputs={
                'in_files': (nii2dicom, 'out_file')},
            outputs={
                'umap_aligned_dicoms': ('out_dir', _format)})
                ,  #  copy2dir parameter
                ,  # internal copy2dir
                ,  # output copy2dir

        return pipeline

    def umap_realignment_pipeline(self, **kwargs):
#         inputs = [FilesetSpec('average_mats', directory_format),
#                   FilesetSpec('umap_ref_coreg_matrix', text_matrix_format),
#                   FilesetSpec('umap_ref_qform_mat', text_matrix_format)]
#         outputs = []
#         if ('umap_ref' in self.sub_study_names and
#                 'umap' in self.input_names):
#             inputs.append(FilesetSpec('umap', nifti_gz_format))
#             outputs.append(FilesetSpec('umaps_align2ref', directory_format))

        pipeline = self.new_pipeline(
            name='umap_realignment',
            desc=("Pipeline to align the original umap (if provided)"
                  "to match the head position in each frame and improve the "
                  "static PET image quality."),
            citations=[fsl_cite],
            **kwargs)

        frame_align = pipeline.add(
            'umap2ref_alignment',
            UmapAlign2Reference(),
            inputs={
                },
            outputs={
                },
            requirements=[fsl_req.v('5.0.9')])
                pct=self.parameter('align_pct'),  #  frame_align parameter
                'ute_regmat': ('umap_ref_coreg_matrix', _format),  # input frame_align
                'ute_qform_mat': ('umap_ref_qform_mat', _format),  # input frame_align
                'average_mats': ('average_mats', _format),  # input frame_align
                'umap': ('umap', _format),  # input frame_align
                'umaps_align2ref': ('umaps_align2ref', _format),  # output frame_align

        return pipeline

    def create_moco_series_pipeline(self, **kwargs):
        """This pipeline is probably wrong as we still do not know how to
        import back the new moco series into the scanner. This was just a first
        attempt.
        """

#             inputs=[FilesetSpec('start_times', text_format),
#                     FilesetSpec('motion_par', text_format)],
#             outputs=[FilesetSpec('moco_series', directory_format)],

        pipeline = self.new_pipeline(
            name='create_moco_series',
            desc=("Pipeline to generate a moco_series that can be then "
                  "imported back in the scanner and used to correct the"
                  " pet data"),
            citations=[fsl_cite],
            **kwargs)

        moco = pipeline.add(
            'create_moco_series',
            CreateMocoSeries(),
            inputs={
                },
            outputs={
                })
                'start_times': ('start_times', _format),  # input moco
                'motion_par': ('motion_par', _format),  # input moco
                moco_template=self.parameter('moco_template'),  #  moco parameter
                'moco_series': ('modified_moco', _format),  # output moco

        return pipeline

    def gather_outputs_pipeline(self, **kwargs):
#         inputs = [FilesetSpec('mean_displacement_plot', png_format),
#                   FilesetSpec('motion_par', text_format),
#                   FilesetSpec('correction_factors', text_format),
#                   FilesetSpec('severe_motion_detection_report', text_format),
#                   FilesetSpec('timestamps', directory_format)]
#         if ('umap_ref' in self.sub_study_names and
#                 'umap' in self.input_names):
#             inputs.append(FilesetSpec('umap_ref_preproc', nifti_gz_format))
#             inputs.append(
#                 FilesetSpec('umap_aligned_dicoms', directory_format))

#             inputs=inputs,
#             outputs=[FilesetSpec('motion_detection_output', directory_format)],

        pipeline = self.new_pipeline(
            name='gather_motion_detection_outputs',

            desc=("Pipeline to gather together all the outputs from "
                  "the motion detection pipeline."),
            citations=[fsl_cite],
            **kwargs)

        merge_inputs = pipeline.add(
            'merge_inputs',
            Merge(5),
            inputs={
                'in1': ('mean_displacement_plot', png_format),
                'in2': ('motion_par', text_format),
                'in3': ('correction_factors', text_format),
                'in4': ('severe_motion_detection_report', text_format),
                'in5': ('timestamps', directory_format)})

        copy2dir = pipeline.add(
            'copy2dir',
            CopyToDir(),
            inputs={
                },
            outputs={
                })
                'in_files': (merge_inputs, 'out'),  # internal copy2dir
                'motion_detection_output': ('out_dir', _format),  # output copy2dir
        return pipeline

    prepare_pet_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_data_preparation_pipeline')

    pet_header_extraction_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_time_info_extraction_pipeline')

    def motion_correction_pipeline(self, **kwargs):
        inputs = [FilesetSpec('pet_data_prepared', directory_format),
                  FilesetSpec('ref_brain', nifti_gz_format),
                  FilesetSpec('mean_displacement_plot', png_format)]
        if self.parameter_spec('dynamic_pet_mc').value:
            inputs.append(FilesetSpec('fixed_binning_mats', directory_format))
            outputs = [FilesetSpec('dynamic_motion_correction_results',
                                   directory_format)]
            dynamic = True
        else:
            inputs.append(FilesetSpec('average_mats', directory_format))
            inputs.append(FilesetSpec('correction_factors', text_format))
            outputs = [FilesetSpec('static_motion_correction_results',
                                   directory_format)]
            dynamic = False
        if 'struct2align' in self.input_names:
            inputs.append(FilesetSpec('struct2align', nifti_gz_format))
            StructAlignment = True
        else:
            StructAlignment = False

        pipeline = self.new_pipeline(
            name='pet_mc',
            inputs=inputs,
            outputs=outputs,
            desc=("Given a folder with reconstructed PET data, this "
                  "pipeline will generate a motion corrected PET"
                  "image using information extracted from the MR-based "
                  "motion detection pipeline"),
            citations=[fsl_cite],
            **kwargs)

        check_pet = pipeline.add(
            'check_pet_data',
            CheckPetMCInputs(),
            inputs={
                },
            outputs={
                },
            requirements=[fsl_req.v('5.0.9'), mrtrix_req.v('3.0rc3')])
                'pet_data': ('pet_data_prepared', _format),  # input check_pet
                'reference': ('ref_brain', _format),  # input check_pet
        if dynamic:
            pipeline.connect_input('fixed_binning_mats', check_pet,
                                   'motion_mats')
        else:
            pipeline.connect_input('average_mats', check_pet,
                                   'motion_mats')
            pipeline.connect_input('correction_factors', check_pet,
                                   'corr_factors')

        if StructAlignment:
            struct_reg = pipeline.add(
                'ref2structural_reg',
                FLIRT(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')])
                    'reference': ('ref_brain', _format),  # input struct_reg
                    'in_file': ('struct2align', _format),  # input struct_reg
                    dof=6,  #  struct_reg parameter
                    cost_func='normmi',  #  struct_reg parameter
                    cost='normmi',  #  struct_reg parameter

        if not dynamic:
            pet_mc = pipeline.add(
                'pet_mc',
                PetImageMotionCorrection(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')],
                iterfield=['corr_factor', 'pet_image', 'motion_mat'])
                    'corr_factor': (check_pet, 'corr_factors'),  # internal pet_mc
        else:
            pet_mc = pipeline.add(
                'pet_mc',
                PetImageMotionCorrection(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')],
                iterfield=['pet_image', 'motion_mat'])
                'pet_image': (check_pet, 'pet_images'),  # internal pet_mc
                'motion_mat': (check_pet, 'motion_mats'),  # internal pet_mc
                'pet2ref_mat': (check_pet, 'pet2ref_mat'),  # internal pet_mc
        if StructAlignment:
            pipeline.connect(struct_reg, 'out_matrix_file', pet_mc,
                             'structural2ref_regmat')
            pipeline.connect_input('struct2align', pet_mc,
                                   'structural_image')
        if self.parameter('PET2MNI_reg'):
            mni_reg = True
        else:
            mni_reg = False

        if dynamic:
            merge_mc = pipeline.add(
                'merge_pet_mc',
                fsl.Merge(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')])
                    dimension='t',  #  merge_mc parameter

            merge_no_mc = pipeline.add(
                'merge_pet_no_mc',
                fsl.Merge(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')])
                    dimension='t',  #  merge_no_mc parameter
                    'in_files': (pet_mc, 'pet_mc_image'),  # internal merge_mc
                    'in_files': (pet_mc, 'pet_no_mc_image'),  # internal merge_no_mc
        else:
            static_mc = pipeline.add(
                'static_mc_generation',
                StaticPETImageGeneration(),
                inputs={
                    },
                outputs={
                    },
                requirements=[fsl_req.v('5.0.9')])
                    'pet_mc_images': (pet_mc, 'pet_mc_image'),  # internal static_mc
                    'pet_no_mc_images': (pet_mc, 'pet_no_mc_image'),  # internal static_mc

        merge_outputs = pipeline.add(
            'merge_outputs',
            Merge(3))
                'in1': ('mean_displacement_plot', _format),  # input merge_outputs

        if not StructAlignment:
            cropping = pipeline.add(
                'pet_cropping',
                PETFovCropping(),
                inputs={
                    },
                outputs={
                    })
                    x_min=self.parameter('crop_xmin'),  #  cropping parameter
                    x_size=self.parameter('crop_xsize'),  #  cropping parameter
                    y_min=self.parameter('crop_ymin'),  #  cropping parameter
                    y_size=self.parameter('crop_ysize'),  #  cropping parameter
                    z_min=self.parameter('crop_zmin'),  #  cropping parameter
                    z_size=self.parameter('crop_zsize'),  #  cropping parameter
            if dynamic:
                pipeline.connect(merge_mc, 'merged_file', cropping,
                                 'pet_image')
            else:
                pipeline.connect(static_mc, 'static_mc', cropping,
                                 'pet_image')

            cropping_no_mc = pipeline.add(
                'pet_no_mc_cropping',
                PETFovCropping(),
                inputs={
                    },
                outputs={
                    })
                    x_min=self.parameter('crop_xmin'),  #  cropping_no_mc parameter
                    x_size=self.parameter('crop_xsize'),  #  cropping_no_mc parameter
                    y_min=self.parameter('crop_ymin'),  #  cropping_no_mc parameter
                    y_size=self.parameter('crop_ysize'),  #  cropping_no_mc parameter
                    z_min=self.parameter('crop_zmin'),  #  cropping_no_mc parameter
                    z_size=self.parameter('crop_zsize'),  #  cropping_no_mc parameter
            if dynamic:
                pipeline.connect(merge_no_mc, 'merged_file', cropping_no_mc,
                                 'pet_image')
            else:
                pipeline.connect(static_mc, 'static_no_mc', cropping_no_mc,
                                 'pet_image')

            if mni_reg:
                if dynamic:
                    t_mean = pipeline.add(
                        'PET_temporal_mean',
                        ImageMaths(),
                        inputs={
                            },
                        outputs={
                            },
                        requirements=[fsl_req.v('5.0.9')])
                            op_string='-Tmean',  #  t_mean parameter
                            'in_file': (cropping, 'pet_cropped'),  # internal t_mean

                reg_tmean2MNI = pipeline.add(
                    'reg2MNI',
                    AntsRegSyn(
                        num_dimensions=3,
                        transformation='s',
                        out_prefix='reg2MNI',
                        num_threads=4),
                    inputs={
                        },
                    outputs={
                        },
                    wall_time=25,
                    requirements=[ants_req.v('2')])
                        ref_file=self.parameter('PET_template_MNI'),  #  reg_tmean2MNI parameter

                if dynamic:
                    pipeline.connect(t_mean, 'out_file', reg_tmean2MNI,
                                     'input_file')

                    merge_trans = pipeline.add(
                        'merge_transforms',
                        Merge(2),
                        inputs={
                            },
                        outputs={
                            },
                        wall_time=1)
                            'in1': (reg_tmean2MNI, 'warp_file'),  # internal merge_trans
                            'in2': (reg_tmean2MNI, 'regmat'),  # internal merge_trans

                    apply_trans = pipeline.add(
                        'apply_trans',
                        ApplyTransforms(),
                        inputs={
                            },
                        outputs={
                            },
                        wall_time=7,
                        mem_gb=24,
                        requirements=[ants_req.v('2')])
                            reference_image=self.parameter('PET_template_MNI'),  #  apply_trans parameter
                            interpolation='Linear',  #  apply_trans parameter
                            input_image_type=3,  #  apply_trans parameter
                            'input_image': (cropping, 'pet_cropped'),  # internal apply_trans
                            'transforms': (merge_trans, 'out'),  # internal apply_trans
                            'in2': (apply_trans, 'output_image'),  # internal merge_outputs
                else:
                    pipeline.connect(cropping, 'pet_cropped', reg_tmean2MNI,
                                     'input_file')
                    pipeline.connect(reg_tmean2MNI, 'reg_file',
                                     merge_outputs, 'in2')
            else:
                        'in2': (cropping, 'pet_cropped'),  # internal merge_outputs
            pipeline.connect(cropping_no_mc, 'pet_cropped', merge_outputs,
                             'in3')
        else:
            if dynamic:
                pipeline.connect(merge_mc, 'merged_file', merge_outputs,
                                 'in2')
                pipeline.connect(merge_no_mc, 'merged_file', merge_outputs,
                                 'in3')
            else:
                pipeline.connect(static_mc, 'static_mc', merge_outputs,
                                 'in2')
                pipeline.connect(static_mc, 'static_no_mc', merge_outputs,
                                 'in3')
#         mcflirt = pipeline.add('mcflirt', MCFLIRT())
#                 'in_file': (merge_mc_ps, 'merged_file'),  # internal mcflirt
#                 cost='normmi',  #  mcflirt parameter

        copy2dir = pipeline.add(
            'copy2dir',
            CopyToDir(),
            inputs={
                },
            outputs={
                })
                'in_files': (merge_outputs, 'out'),  # internal copy2dir
        if dynamic:
            pipeline.connect_output('dynamic_motion_correction_results',
                                    copy2dir, 'out_dir')
        else:
            pipeline.connect_output('static_motion_correction_results',
                                    copy2dir, 'out_dir')
        return pipeline


def create_motion_correction_class(name, ref=None, ref_type=None, t1s=None,
                                   t2s=None, dwis=None, epis=None,
                                   umap=None, dynamic=False, umap_ref=None,
                                   pet_data_dir=None, pet_recon_dir=None,
                                   struct2align=None):

    inputs = []
    dct = {}
    data_specs = []
    run_pipeline = False
    parameter_specs = [ParameterSpec('ref_preproc_resolution', [1])]
    switch_specs = []
    if struct2align is not None:
        struct_image = struct2align.split('/')[-1].split('.')[0]

    if pet_data_dir is not None:
        inputs.append(FilesetSelector('pet_data_dir',
                                      'pet_data_dir', directory_format))
    if pet_recon_dir is not None:
        inputs.append(FilesetSelector('pet_data_reconstructed',
                                      'pet_data_reconstructed',
                                      directory_format))
        if struct2align is not None:
            inputs.append(
                FilesetSelector('struct2align', struct_image, nifti_gz_format))
    if pet_data_dir is not None and pet_recon_dir is not None and dynamic:
        output_data = 'dynamic_motion_correction_results'
        parameter_specs.append(ParameterSpec('dynamic_pet_mc', True))
        if struct2align is not None:
            inputs.append(
                FilesetSelector('struct2align', struct_image, nifti_gz_format))
    elif (pet_recon_dir is not None and not dynamic):
        output_data = 'static_motion_correction_results'
    else:
        output_data = 'motion_detection_output'

    if not ref:
        raise Exception('A reference image must be provided!')
    if ref_type == 't1':
        ref_study = T1Study
    elif ref_type == 't2':
        ref_study = T2Study
    else:
        raise Exception('{} is not a recognized ref_type!The available '
                        'ref_types are t1 or t2.'.format(ref_type))

    study_specs = [SubStudySpec('ref', ref_study)]
    ref_spec = {'ref_brain': 'coreg_ref_brain'}
    inputs.append(FilesetSelector('ref_primary', ref, dicom_format))

    if umap_ref and umap:
        if umap_ref.endswith('/'):
            umap_ref = umap_ref.split('/')[-2]
        else:
            umap_ref = umap_ref.split('/')[-1]
        if umap_ref in t1s:
            umap_ref_study = T1Study
            t1s.remove(umap_ref)
        elif umap_ref in t2s:
            umap_ref_study = T2Study
            t2s.remove(umap_ref)
        else:
            umap_ref = None

    if t1s:
        study_specs.extend([SubStudySpec('t1_{}'.format(i), T1Study,
                                         ref_spec) for i in range(len(t1s))])
        inputs.extend(
            FilesetSelector('t1_{}_primary'.format(i), dicom_format, t1_scan)
            for i, t1_scan in enumerate(t1s))
        run_pipeline = True

    if t2s:
        study_specs.extend([SubStudySpec('t2_{}'.format(i), T2Study,
                                         ref_spec) for i in range(len(t2s))])
        inputs.extend(FilesetSelector('t2_{}_primary'.format(i),
                                      t2_scan, dicom_format)
                      for i, t2_scan in enumerate(t2s))
        run_pipeline = True

    if umap_ref and not umap:
        logger.info('Umap not provided. The umap realignment will not be '
                    'performed. Umap_ref will be trated as {}'
                    .format(umap_ref_study))

    elif umap_ref and umap:
        logger.info('Umap will be realigned to match the head position in '
                    'each frame.')
        if type(umap) == list and len(umap) > 1:
            logger.info('More than one umap provided. Only the first one will '
                        'be used.')
            umap = umap[0]
        study_specs.append(SubStudySpec('umap_ref', umap_ref_study, ref_spec))
        inputs.append(FilesetSelector('umap_ref_primary', dicom_format,
                                      umap_ref))
        inputs.append(FilesetSelector('umap', dicom_format, umap))

        run_pipeline = True

    elif not umap_ref and umap:
        logger.warning('Umap provided without corresponding reference image. '
                       'Realignment cannot be performed without umap_ref. Umap'
                       ' will be ignored.')

    if epis:
        epi_refspec = ref_spec.copy()
        epi_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                            'ref_preproc': 'coreg_ref'})
        study_specs.extend(SubStudySpec('epi_{}'.format(i), EpiStudy,
                                        epi_refspec)
                           for i in range(len(epis)))
        inputs.extend(
            FilesetSelector('epi_{}_primary'.format(i), epi_scan, dicom_format)
            for i, epi_scan in enumerate(epis))
        run_pipeline = True
    if dwis:
        unused_dwi = []
        dwis_main = [x for x in dwis if x[-1] == '0']
        dwis_ref = [x for x in dwis if x[-1] == '1']
        dwis_opposite = [x for x in dwis if x[-1] == '-1']
        dwi_refspec = ref_spec.copy()
        dwi_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                           'ref_preproc': 'coreg_ref'})
        if dwis_main:
            switch_specs.extend(
                SwitchSpec('dwi_{}_brain_extract_method'.format(i), 'fsl',
                           ('mrtrix', 'fsl')) for i in range(len(dwis_main)))
        if dwis_main and not dwis_opposite:
            logger.warning(
                'No opposite phase encoding direction b0 provided. DWI '
                'motion correction will be performed without distortion '
                'correction. THIS IS SUB-OPTIMAL!')
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DwiStudy, dwi_refspec)
                for i in range(len(dwis_main)))
            inputs.extend(
                FilesetSelector('dwi_{}_primary'.format(i),
                                dwis_main_scan[0], dicom_format)
                for i, dwis_main_scan in enumerate(dwis_main))
        if dwis_main and dwis_opposite:
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DwiStudy, dwi_refspec)
                for i in range(len(dwis_main)))
            inputs.extend(
                FilesetSelector(
                    'dwi_{}_primary'.format(i),
                    dwis_main[i][0],
                    dicom_format)
                for i in range(len(dwis_main)))
            if len(dwis_main) <= len(dwis_opposite):
                inputs.extend(
                    FilesetSelector('dwi_{}_dwi_reference'.format(i),
                                    dwis_opposite[i][0], dicom_format)
                    for i in range(len(dwis_main)))
            else:
                inputs.extend(FilesetSelector('dwi_{}_dwi_reference'.format(i),
                                              dwis_opposite[0][0],
                                              dicom_format)
                              for i in range(len(dwis_main)))
        if dwis_opposite and dwis_main and not dwis_ref:
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EpiStudy, dwi_refspec)
                for i in range(len(dwis_opposite)))
            inputs.extend(FilesetSelector('b0_{}_primary'.format(i),
                                          dwis_opposite[i][0], dicom_format)
                          for i in range(len(dwis_opposite)))
            if len(dwis_opposite) <= len(dwis_main):
                inputs.extend(FilesetSelector('b0_{}_reverse_phase'.format(i),
                                              dwis_main[i][0], dicom_format)
                              for i in range(len(dwis_opposite)))
            else:
                inputs.extend(FilesetSelector('b0_{}_reverse_phase'.format(i),
                                              dwis_main[0][0], dicom_format)
                              for i in range(len(dwis_opposite)))
        elif dwis_opposite and dwis_ref:
            min_index = min(len(dwis_opposite), len(dwis_ref))
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EpiStudy, dwi_refspec)
                for i in range(min_index * 2))
            inputs.extend(
                FilesetSelector('b0_{}_primary'.format(i), scan[0],
                                dicom_format)
                for i, scan in enumerate(dwis_opposite[:min_index] +
                                         dwis_ref[:min_index]))
            inputs.extend(
                FilesetSelector('b0_{}_reverse_phase'.format(i), scan[0],
                                dicom_format)
                for i, scan in enumerate(dwis_ref[:min_index] +
                                         dwis_opposite[:min_index]))
            unused_dwi = [scan for scan in dwis_ref[min_index:] +
                          dwis_opposite[min_index:]]
        elif dwis_opposite or dwis_ref:
            unused_dwi = [scan for scan in dwis_ref + dwis_opposite]
        if unused_dwi:
            logger.info(
                'The following scans:\n{}\nwere not assigned during the DWI '
                'motion detection initialization (probably a different number '
                'of main DWI scans and b0 images was provided). They will be '
                'processed os "other" scans.'
                .format('\n'.join(s[0] for s in unused_dwi)))
            study_specs.extend(
                SubStudySpec('t2_{}'.format(i), T2Study, ref_spec)
                for i in range(len(t2s), len(t2s) + len(unused_dwi)))
            inputs.extend(
                FilesetSelector('t2_{}_primary'.format(i), scan[0],
                                dicom_format)
                for i, scan in enumerate(unused_dwi, start=len(t2s)))
        run_pipeline = True

    if not run_pipeline:
        raise Exception('At least one scan, other than the reference, must be '
                        'provided!')

    dct['add_sub_study_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['__metaclass__'] = MultiStudyMetaClass
    dct['add_parameter_specs'] = parameter_specs
    dct['add_switch_specs'] = switch_specs
    return (MultiStudyMetaClass(name, (MotionDetectionMixin,), dct), inputs,
            output_data)


def create_motion_detection_class(name, ref=None, ref_type=None, t1s=None,
                                  t2s=None, dwis=None, epis=None,
                                  pet_data_dir=None):

    inputs = []
    dct = {}
    data_specs = []
    run_pipeline = False
    parameter_specs = [ParameterSpec('ref_preproc_resolution', [1])]

    if pet_data_dir is not None:
        inputs.append(FilesetSelector('pet_data_dir', 'pet_data_dir',
                                      directory_format))

    if not ref:
        raise Exception('A reference image must be provided!')
    if ref_type == 't1':
        ref_study = T1Study
    elif ref_type == 't2':
        ref_study = T2Study
    else:
        raise Exception('{} is not a recognized ref_type!The available '
                        'ref_types are t1 or t2.'.format(ref_type))

    study_specs = [SubStudySpec('ref', ref_study)]
    ref_spec = {'ref_brain': 'coreg_ref_brain'}
    inputs.append(FilesetSelector('ref_primary', ref, dicom_format))

    if t1s:
        study_specs.extend(
            [SubStudySpec('t1_{}'.format(i), T1Study, ref_spec)
             for i in range(len(t1s))])
        inputs.extend(
            FilesetSelector('t1_{}_primary'.format(i), t1_scan, dicom_format)
            for i, t1_scan in enumerate(t1s))
        run_pipeline = True

    if t2s:
        study_specs.extend(
            [SubStudySpec('t2_{}'.format(i), T2Study, ref_spec)
             for i in range(len(t2s))])
        inputs.extend(FilesetSelector('t2_{}_primary'.format(i), t2_scan,
                                      dicom_format)
                      for i, t2_scan in enumerate(t2s))
        run_pipeline = True

    if epis:
        epi_refspec = ref_spec.copy()
        epi_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                            'ref_preproc': 'coreg_ref'})
        study_specs.extend(SubStudySpec('epi_{}'.format(i), EpiStudy,
                                        epi_refspec)
                           for i in range(len(epis)))
        inputs.extend(
            FilesetSelector('epi_{}_primary'.format(i), epi_scan, dicom_format)
            for i, epi_scan in enumerate(epis))
        run_pipeline = True
    if dwis:
        unused_dwi = []
        dwis_main = [x for x in dwis if x[-1] == '0']
        dwis_ref = [x for x in dwis if x[-1] == '1']
        dwis_opposite = [x for x in dwis if x[-1] == '-1']
        b0_refspec = ref_spec.copy()
        b0_refspec.update({'ref_wm_seg': 'coreg_ref_wmseg',
                           'ref_preproc': 'coreg_ref'})
        if dwis_main and not dwis_opposite:
            logger.warning(
                'No opposite phase encoding direction b0 provided. DWI '
                'motion correction will be performed without distortion '
                'correction. THIS IS SUB-OPTIMAL!')
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DwiStudy, ref_spec)
                for i in range(len(dwis_main)))
            inputs.extend(
                FilesetSelector('dwi_{}_primary'.format(i),
                                dwis_main_scan[0], dicom_format)
                for i, dwis_main_scan in enumerate(dwis_main))
        if dwis_main and dwis_opposite:
            study_specs.extend(
                SubStudySpec('dwi_{}'.format(i), DwiStudy, ref_spec)
                for i in range(len(dwis_main)))
            inputs.extend(
                FilesetSelector(
                    'dwi_{}_primary'.format(i), dwis_main[i][0], dicom_format)
                for i in range(len(dwis_main)))
            if len(dwis_main) <= len(dwis_opposite):
                inputs.extend(FilesetSelector('dwi_{}_dwi_reference'.format(i),
                                              dwis_opposite[i][0],
                                              dicom_format)
                              for i in range(len(dwis_main)))
            else:
                inputs.extend(FilesetSelector('dwi_{}_dwi_reference'.format(i),
                                              dwis_opposite[0][0],
                                              dicom_format)
                              for i in range(len(dwis_main)))
        if dwis_opposite and dwis_main and not dwis_ref:
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EpiStudy, b0_refspec)
                for i in range(len(dwis_opposite)))
            inputs.extend(FilesetSelector('b0_{}_primary'.format(i),
                                          dwis_opposite[i][0], dicom_format)
                          for i in range(len(dwis_opposite)))
            if len(dwis_opposite) <= len(dwis_main):
                inputs.extend(FilesetSelector('b0_{}_reverse_phase'.format(i),
                                              dwis_main[i][0], dicom_format)
                              for i in range(len(dwis_opposite)))
            else:
                inputs.extend(FilesetSelector('b0_{}_reverse_phase'.format(i),
                                              dwis_main[0][0], dicom_format)
                              for i in range(len(dwis_opposite)))
        elif dwis_opposite and dwis_ref:
            min_index = min(len(dwis_opposite), len(dwis_ref))
            study_specs.extend(
                SubStudySpec('b0_{}'.format(i), EpiStudy, b0_refspec)
                for i in range(min_index * 2))
            inputs.extend(
                FilesetSelector('b0_{}_primary'.format(i),
                                scan[0], dicom_format)
                for i, scan in enumerate(dwis_opposite[:min_index] +
                                         dwis_ref[:min_index]))
            inputs.extend(
                FilesetSelector('b0_{}_reverse_phase'.format(i),
                                scan[0], dicom_format)
                for i, scan in enumerate(dwis_ref[:min_index] +
                                         dwis_opposite[:min_index]))
            unused_dwi = [scan for scan in dwis_ref[min_index:] +
                          dwis_opposite[min_index:]]
        elif dwis_opposite or dwis_ref:
            unused_dwi = [scan for scan in dwis_ref + dwis_opposite]
        if unused_dwi:
            logger.info(
                'The following scans:\n{}\nwere not assigned during the DWI '
                'motion detection initialization (probably a different number '
                'of main DWI scans and b0 images was provided). They will be '
                'processed os "other" scans.'
                .format('\n'.join(s[0] for s in unused_dwi)))
            study_specs.extend(
                SubStudySpec('t2_{}'.format(i), T2Study, ref_spec)
                for i in range(len(t2s), len(t2s) + len(unused_dwi)))
            inputs.extend(
                FilesetSelector('t2_{}_primary'.format(i), scan[0],
                                dicom_format)
                for i, scan in enumerate(unused_dwi, start=len(t2s)))
        run_pipeline = True

    if not run_pipeline:
        raise Exception('At least one scan, other than the reference, must be '
                        'provided!')

    dct['add_sub_study_specs'] = study_specs
    dct['add_data_specs'] = data_specs
    dct['__metaclass__'] = MultiStudyMetaClass
    dct['add_param_specs'] = parameter_specs
    return MultiStudyMetaClass(name, (MotionDetectionMixin,), dct), inputs
