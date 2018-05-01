from nianalysis.dataset import DatasetSpec, FieldSpec, Field
from nianalysis.data_format import (
    nifti_gz_format, text_matrix_format, directory_format, text_format,
    png_format, dicom_format)
from mbianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming, PlotMeanDisplacementRC,
    AffineMatAveraging, PetCorrectionFactor, FrameAlign2Reference)
from mbianalysis.citation import fsl_cite
from nianalysis.study.base import StudyMetaClass
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from mbianalysis.study.mri.epi import CoregisteredEPIStudy
from mbianalysis.study.mri.structural.t1 import CoregisteredT1Study, T1Study
from mbianalysis.study.mri.structural.t2 import CoregisteredT2Study, T2Study
from nipype.interfaces.utility import Merge
from mbianalysis.study.mri.structural.diffusion_coreg import (
    CoregisteredDiffusionStudy, CoregisteredDiffusionReferenceOppositeStudy,
    CoregisteredDiffusionReferenceStudy)
from mbianalysis.requirement import fsl509_req
from nianalysis.exception import NiAnalysisNameError
from nianalysis.dataset import Dataset
import logging
from mbianalysis.study.mri.structural.ute import CoregisteredUTEStudy
from nianalysis.interfaces.utils import CopyToDir
from mbianalysis.study.pet.base import PETStudy
from mbianalysis.interfaces.custom.pet import StaticMotionCorrection
from nianalysis.option import OptionSpec


logger = logging.getLogger('NiAnalysis')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("urllib3").setLevel(logging.WARNING)


class MotionReferenceT1Study(T1Study):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('wm_seg', nifti_gz_format, 'segmentation_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'header_info_extraction_pipeline'),
        FieldSpec('tr', dtype=float, pipeline=header_info_extraction_pipeline),
        FieldSpec('start_time', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('real_duration', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('tot_duration', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('ped', str, pipeline=header_info_extraction_pipeline),
        FieldSpec('pe_angle', str,
                  pipeline=header_info_extraction_pipeline),
        DatasetSpec(
            'dcm_info',
            text_format,
            'header_info_extraction_pipeline'),
        DatasetSpec('preproc', nifti_gz_format, 'basic_preproc_pipeline')]

    add_option_specs = [
        OptionSpec('preproc_resolution', [1])]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(MotionReferenceT1Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', ref=True, multivol=False, **kwargs))

    def segmentation_pipeline(self, **kwargs):
        pipeline = super(MotionReferenceT1Study, self).segmentation_pipeline(
            img_type=1, **kwargs)
        return pipeline


class MotionReferenceT2Study(T2Study):

    __metaclass__ = StudyMetaClass

    add_data_specs = [
        DatasetSpec('wm_seg', nifti_gz_format, 'segmentation_pipeline'),
        DatasetSpec('motion_mats', directory_format,
                    'header_info_extraction_pipeline'),
        FieldSpec('tr', dtype=float, pipeline=header_info_extraction_pipeline),
        FieldSpec('start_time', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('real_duration', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('tot_duration', str,
                  pipeline=header_info_extraction_pipeline),
        FieldSpec('ped', str, pipeline=header_info_extraction_pipeline),
        FieldSpec('pe_angle', str,
                  pipeline=header_info_extraction_pipeline),
        DatasetSpec(
            'dcm_info',
            text_format,
            'header_info_extraction_pipeline'),
        DatasetSpec('preproc', nifti_gz_format, 'basic_preproc_pipeline')]

    add_option_specs = [
        OptionSpec('preproc_resolution', [1])]

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(MotionReferenceT2Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', ref=True, multivol=False, **kwargs))

    def segmentation_pipeline(self, **kwargs):
        pipeline = super(MotionReferenceT2Study, self).segmentation_pipeline(
            img_type=2, **kwargs)
        return pipeline


class MotionCorrectionMixin(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

#     ute_brain_mask_pipeline = MultiStudy.translate(
#         'ute', 't1_brain_mask_pipeline',
#         override_default_options={'bet_method': 'optibet'})

    add_sub_study_specs = [
        SubStudySpec('pet_mc', PETStudy, {
            'pet_data_dir': 'pet_data_dir',
            'pet_data_reconstructed': 'pet_recon_dir',
            'pet_data_prepared': 'pet_recon_dir_prepared',
            'pet_data_cropped': 'pet_data_cropped',
            'static_pet_mc2crop': 'pet2crop'})]

    add_data_specs = [
        DatasetSpec('pet_data_dir', directory_format),
        DatasetSpec('pet_data_reconstructed', directory_format),
        DatasetSpec('pet_data_prepared', directory_format,
                    'prepare_pet_pipeline'),
        DatasetSpec('pet_data_cropped', directory_format,
                    'pet_fov_cropping_pipeline'),
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
                    'frame2ref_alignment_pipeline'),
        DatasetSpec('frame2reference_mats', directory_format,
                    'frame2ref_alignment_pipeline'),
        DatasetSpec('motion_detection_output', directory_format,
                    'gather_outputs_pipeline'),
        DatasetSpec('static_pet_mc', directory_format,
                    'static_motion_correction_pipeline'),
        DatasetSpec('static_pet_mc2crop', directory_format,
                    'static_motion_correction_pipeline'),
        FieldSpec('pet_duration', dtype=int,
                  pipeline='pet_time_info_extraction_pipeline'),
        FieldSpec('pet_end_time', dtype=str,
                  pipeline='pet_time_info_extraction_pipeline'),
        FieldSpec('pet_start_time', dtype=str,
                  pipeline='pet_time_info_extraction_pipeline')]

    add_option_specs = [OptionSpec('framing_th', 2.0),
                        OptionSpec('framing_temporal_th', 30.0),
                        OptionSpec('md_framing', True),
                        OptionSpec('align_pct', False),
                        OptionSpec('align_fixed_binning', False)]

    def mean_displacement_pipeline(self, **kwargs):
        inputs = [DatasetSpec('ref_masked', nifti_gz_format)]
        sub_study_names = []
        for sub_study_spec in self.sub_study_specs():
            try:
                inputs.append(
                    self.dataset(sub_study_spec.inverse_map('motion_mats')))
                inputs.append(self.dataset(sub_study_spec.inverse_map('tr')))
                inputs.append(
                    self.dataset(sub_study_spec.inverse_map('start_time')))
                inputs.append(
                    self.dataset(sub_study_spec.inverse_map('real_duration')))
                sub_study_names.append(sub_study_spec.name)
            except NiAnalysisNameError:
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
                     DatasetSpec('mats4average', text_format)],
            description=("Calculate the mean displacement between each motion"
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
        pipeline.connect(merge_motion_mats, 'out', md, 'motion_mats')
        pipeline.connect(merge_tr, 'out', md, 'trs')
        pipeline.connect(merge_start_time, 'out', md, 'start_times')
        pipeline.connect(merge_real_duration, 'out', md, 'real_durations')
        pipeline.connect_input('ref_masked', md, 'reference')
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
        return pipeline

    def motion_framing_pipeline(self, **kwargs):
        return self.motion_framing_pipeline_factory(
            pet_data_dir=None, pet_start_time=None, pet_duration=None,
            **kwargs)

    def motion_framing_pipeline_factory(
            self, pet_data_dir=None, pet_start_time=None, pet_duration=None,
            **kwargs):
        inputs = [DatasetSpec('mean_displacement', text_format),
                  DatasetSpec('mean_displacement_consecutive', text_format),
                  DatasetSpec('start_times', text_format)]
        if pet_data_dir is not None:
            inputs.append(DatasetSpec(pet_data_dir, directory_format))
        elif pet_start_time is not None and pet_duration is not None:
            inputs.append(FieldSpec(pet_start_time, str))
            inputs.append(FieldSpec(pet_duration, str))
        pipeline = self.create_pipeline(
            name='motion_framing',
            inputs=inputs,
            outputs=[DatasetSpec('frame_start_times', text_format),
                     DatasetSpec('frame_vol_numbers', text_format),
                     DatasetSpec('timestamps', directory_format)],
            description=("Calculate when the head movement exceeded a "
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
        if pet_data_dir is not None:
            pipeline.connect_input('pet_data_dir', framing, 'pet_data_dir')
        elif pet_start_time is not None and pet_duration is not None:
            pipeline.connect_input('pet_start_time', framing, 'pet_start_time')
            pipeline.connect_input('pet_duration', framing, 'pet_duration')
#         else:
#             framing.inputs.pet_data_dir = None
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
            description=("Plot the mean displacement real clock"),
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
            description=("Average all the transformation mats within each "
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
            description=("Pipeline to calculate the correction factors to "
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

    def frame2ref_alignment_pipeline_factory(
            self, name, average_mats, ute_regmat, ute_qform_mat, umap=None,
            **kwargs):
        inputs = [DatasetSpec(average_mats, directory_format),
                  DatasetSpec(ute_regmat, text_matrix_format),
                  DatasetSpec(ute_qform_mat, text_matrix_format)]
        outputs = [DatasetSpec('frame2reference_mats', directory_format)]
        if umap:
            inputs.append(DatasetSpec(umap, nifti_gz_format))
            outputs.append(DatasetSpec('umaps_align2ref', directory_format))

        pipeline = self.create_pipeline(
            name=name,
            inputs=inputs,
            outputs=outputs,
            description=("Pipeline to create an affine mat to align each "
                         "detected frame to the reference. If umap is provided"
                         ", it will be also aligned to match the head position"
                         " in each frame and improve the static PET image "
                         "quality."),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        frame_align = pipeline.create_node(
            FrameAlign2Reference(), name='frame2ref_alignment',
            requirements=[fsl509_req])
        frame_align.inputs.pct = pipeline.option('align_pct')
        frame_align.inputs.fixed_binning = pipeline.option(
            'align_fixed_binning')
        pipeline.connect_input(average_mats, frame_align,
                               'average_mats')
        pipeline.connect_input(ute_regmat, frame_align,
                               'ute_regmat')
        pipeline.connect_input(ute_qform_mat, frame_align,
                               'ute_qform_mat')
        if umap:
            pipeline.connect_input(umap, frame_align, 'umap')
            pipeline.connect_output('umaps_align2ref', frame_align,
                                    'umaps_align2ref')
        pipeline.connect_output('frame2reference_mats', frame_align,
                                'frame2reference_mats')
        return pipeline

    def frame2ref_alignment_pipeline(self, **kwargs):
        return self.frame2ref_alignment_pipeline_factory(
            'frame2ref_alignment', 'average_mats', 'ute_reg_mat',
            'ute_qform_mat', umap='umap_nifti',
            **kwargs)

    def gather_outputs_factory(self, name, align_mats=False,
                               pet_corr_fac=False, aligned_umaps=False,
                               timestamps=False, ute=None, **kwargs):
        inputs = [DatasetSpec('mean_displacement_plot', png_format),
                  DatasetSpec('motion_par', text_format)]
        if align_mats:
            inputs.append(
                DatasetSpec('frame2reference_mats', directory_format))
        if pet_corr_fac:
            inputs.append(DatasetSpec('correction_factors', text_format))
        if aligned_umaps:
            inputs.append(
                DatasetSpec('umaps_align2ref_dicom', directory_format))
        if timestamps:
            inputs.append(DatasetSpec('timestamps', directory_format))
        if ute is not None:
            inputs.append(DatasetSpec(ute, nifti_gz_format))

        pipeline = self.create_pipeline(
            name=name,
            inputs=inputs,
            outputs=[DatasetSpec('motion_detection_output', directory_format)],
            description=("Pipeline to gather together all the outputs from "
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

    def gather_outputs_pipeline(self, **kwargs):
        return self.gather_outputs_factory(
            'gather_md_outputs', pet_corr_fac=False, aligned_umaps=False,
            timestamps=False, align_mats=False, **kwargs)

    prepare_pet_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_data_preparation_pipeline')

    pet_fov_cropping_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_fov_cropping_pipeline')

    pet_time_info_extraction_pipeline = MultiStudy.translate(
        'pet_mc', 'pet_time_info_extraction_pipeline')

    def static_motion_correction_pipeline_factory(self, StructAlignment=None,
                                                  **kwargs):
        inputs = [DatasetSpec('pet_data_prepared', directory_format),
                  DatasetSpec('motion_detection_output', directory_format)]
        if StructAlignment is not None:
            inputs.append(DatasetSpec(StructAlignment, nifti_gz_format))

        pipeline = self.create_pipeline(
            name='static_mc',
            inputs=inputs,
            outputs=[DatasetSpec('static_pet_mc', directory_format),
                     DatasetSpec('static_pet_mc2crop', directory_format)],
            description=("Given a folder with reconstructed PET data, this "
                         "pipeline will generate a motion corrected static PET"
                         "image using information extracted from the MR-based "
                         "motion detection pipeline"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        static_mc = pipeline.create_node(
            StaticMotionCorrection(), name='static_mc',
            requirements=[fsl509_req])
        pipeline.connect_input('pet_data_prepared', static_mc, 'pet_cropped')
        pipeline.connect_input('motion_detection_output', static_mc, 'md_dir')
        if StructAlignment is not None:
            pipeline.connect_input(StructAlignment, static_mc,
                                   'structural_image')
        pipeline.connect_output('static_pet_mc', static_mc, 'pet_mc_results')
        pipeline.connect_output('static_pet_mc2crop', static_mc,
                                'pet_mc_results_to_crop')
        return pipeline

    def static_motion_correction_pipeline(self, **kwargs):
        return self.static_motion_correction_pipeline_factory(
            StructAlignment=None, **kwargs)


def create_motion_detection_class(name, ref=None, ref_type=None, t1s=None,
                                  utes=None, t2s=None, dmris=None, epis=None,
                                  umaps=None, dynamic=False,
                                  pet_data_dir=None, pet_time_info=None,
                                  motion_correction=False):
    inputs = {}
    dct = {}
    data_specs = []
    run_pipeline = False
    if motion_correction:
        inputs['pet_data_reconstructed'] = Dataset(
            'pet_data_reconstructed', directory_format)
    if pet_data_dir is not None:
        data_specs.append(Dataset('pet_data_dir', directory_format))
        inputs['pet_data_dir'] = Dataset('pet_data_dir', directory_format)

        def motion_framing_pipeline_altered(self, **kwargs):
            return self.motion_framing_pipeline_factory(
                pet_data_dir='pet_data_dir', **kwargs)

        dct['motion_framing_pipeline'] = motion_framing_pipeline_altered
        data_specs.append(
            DatasetSpec('frame_start_times', text_format,
                        'motion_framing_pipeline_altered'))
        data_specs.append(
            DatasetSpec('frame_vol_numbers', text_format,
                        'motion_framing_pipeline_altered'))
        data_specs.append(
            DatasetSpec('timestamps', directory_format,
                        'motion_framing_pipeline_altered'))
    elif pet_time_info is not None:
        data_specs.append(Field('pet_start_time', str))
        data_specs.append(Field('pet_duration', str))
        inputs['pet_start_time'] = Field(pet_time_info[0], str)
        inputs['pet_duration'] = Field(pet_time_info[1], str)

        def motion_framing_pipeline_altered(self, **kwargs):
            return self.motion_framing_pipeline_factory(
                pet_start_time=pet_time_info[0], pet_duration=pet_time_info[1],
                **kwargs)

        dct['motion_framing_pipeline'] = motion_framing_pipeline_altered
        data_specs.append(
            DatasetSpec('frame_start_times', text_format,
                        'motion_framing_pipeline_altered'))
        data_specs.append(
            DatasetSpec('frame_vol_numbers', text_format,
                        'motion_framing_pipeline_altered'))
        data_specs.append(
            DatasetSpec('timestamps', directory_format,
                        'motion_framing_pipeline_altered'))
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
    ref_spec = {'ref_preproc': 'ref_preproc',
                'ref_masked': 'ref_brain',
                'ref_brain_mask': 'ref_brain_mask'}
    inputs['ref_primary'] = Dataset(ref, dicom_format)

    if not utes:
        logger.info('UTE not provided. The matrices that realign the PET image'
                    ' in each detected frame to the reference cannot be '
                    'generated')

        def gather_md_outputs_pipeline_altered(self, **kwargs):
            return self.gather_outputs_factory(
                'gather_md_outputs', pet_corr_fac=True, aligned_umaps=False,
                timestamps=True, align_mats=False)

        dct['gather_outputs_pipeline'] = (
            gather_md_outputs_pipeline_altered)

    elif utes and not umaps:
        logger.info('Umap not provided. The umap realignment will not be '
                    'performed. Matrices that realign each detected frame to '
                    'the reference will be calculated.')

        study_specs.extend(
            [SubStudySpec('ute_{}'.format(i), CoregisteredUTEStudy,
                          ref_spec) for i in range(len(utes))])
        inputs.update({'ute_{}_ute'.format(i): Dataset(ute_scan, dicom_format)
                       for i, ute_scan in enumerate(utes)})

        def frame2ref_alignment_pipeline_altered(self, **kwargs):
            return self.frame2ref_alignment_pipeline_factory(
                'frame2ref_alignment', 'average_mats',
                'ute_{}_ute_reg_mat'.format(len(utes) - 1),
                'ute_{}_ute_qform_mat'.format(len(utes) - 1), umap=None,
                **kwargs)

        def gather_md_outputs_pipeline_altered(self, **kwargs):
            return self.gather_outputs_factory(
                'gather_md_outputs', pet_corr_fac=True, aligned_umaps=False,
                timestamps=True, align_mats=True, **kwargs)

        dct['gather_outputs_pipeline'] = (
            gather_md_outputs_pipeline_altered)
        dct['frame2ref_alignment_pipeline'] = (
            frame2ref_alignment_pipeline_altered)
        data_specs.append(DatasetSpec(
            'frame2reference_mats', directory_format,
            frame2ref_alignment_pipeline_altered))
        run_pipeline = True

    elif utes and umaps:
        logger.info('Umap will be realigned to match the head position in '
                    'each frame. Matrices that realign each frame to the '
                    'reference will be calculated.')
        if len(umaps) > 1:
            raise Exception('Please provide just one umap.')
        ute_spec = ref_spec.copy()
        ute_spec.update({'umaps_align2ref': 'umap_aligned_niftis',
                         'umaps_align2ref_dicom': 'umap_aligned_dicoms'})
        study_specs.extend(
            [SubStudySpec('ute_{}'.format(i), CoregisteredUTEStudy,
                          ute_spec) for i in range(len(utes))])
        inputs.update({'ute_{}_ute'.format(i): Dataset(ute_scan, dicom_format)
                       for i, ute_scan in enumerate(utes)})

        def frame2ref_alignment_pipeline_altered(self, **kwargs):
            return self.frame2ref_alignment_pipeline_factory(
                'frame2ref_alignment', 'average_mats',
                'ute_{}_ute_reg_mat'.format(len(utes) - 1),
                'ute_{}_ute_qform_mat'.format(len(utes) - 1),
                umap='ute_{}_umap_nifti'.format(len(utes) - 1),
                pct=False, fixed_binning=False, **kwargs)

        def gather_md_outputs_pipeline_altered(self, **kwargs):
            return self.gather_outputs_factory(
                'gather_md_outputs', pet_corr_fac=True, aligned_umaps=True,
                timestamps=True, align_mats=True,
                ute='ute_{}_ute_preproc'.format(len(utes) - 1),
                **kwargs)

        dct['gather_outputs_pipeline'] = (
            gather_md_outputs_pipeline_altered)
        dct['frame2ref_alignment_pipeline'] = (
            frame2ref_alignment_pipeline_altered)
        dct['nii2dcm_conversion_pipeline'] = MultiStudy.translate(
            'ute_{}'.format(len(utes) - 1), CoregisteredUTEStudy.
            nifti2dcm_conversion_pipeline)
        inputs['ute_{}_umap'.format(len(utes) - 1)] = Dataset(
            umaps[0], dicom_format)
        data_specs.append(
            DatasetSpec('frame2reference_mats', directory_format,
                        'frame2ref_alignment_pipeline_altered'))
        data_specs.append(
            DatasetSpec('umaps_align2ref', directory_format,
                        'frame2ref_alignment_pipeline_altered'))
        data_specs.append(
            DatasetSpec('umaps_align2ref_dicom', directory_format,
                        dct['nii2dcm_conversion_pipeline']))
        run_pipeline = True

    elif not utes and umaps:
        logger.warning('Umap provided without corresponding UTE image. '
                       'Realignment cannot be performed without UTE. Umap will'
                       'be ignored.')
    if t1s:
        study_specs.extend(
            [SubStudySpec('t1_{}'.format(i), CoregisteredT1Study,
                          ref_spec) for i in range(len(t1s))])
        inputs.update({'t1_{}_t1'.format(i): Dataset(t1_scan, dicom_format)
                       for i, t1_scan in enumerate(t1s)})
        run_pipeline = True

    if t2s:
        study_specs.extend(
            [SubStudySpec('t2_{}'.format(i), CoregisteredT2Study,
                          ref_spec) for i in range(len(t2s))])
        inputs.update({'t2_{}_t2'.format(i): Dataset(t2_scan, dicom_format)
                       for i, t2_scan in enumerate(t2s)})
        run_pipeline = True
    if epis:
        epi_refspec = ref_spec.copy()
        epi_refspec.update({'ref_wm_seg': 'ref_wmseg'})
        study_specs.extend([SubStudySpec('epi_{}'.format(i),
                                         CoregisteredEPIStudy,
                                         epi_refspec)
                            for i in range(len(epis))])
        inputs.update({'epi_{}_epi'.format(i): Dataset(epi_scan, dicom_format)
                       for i, epi_scan in enumerate(epis)})
        run_pipeline = True
    if dmris:
        dmris_main = [x for x in dmris if x[-1] == '0']
        dmris_ref = [x for x in dmris if x[-1] == '1']
        dmris_opposite = [x for x in dmris if x[-1] == '-1']
        if dmris_main and not dmris_opposite:
            raise Exception('If you provide one main Diffusion image you '
                            'have also to provide an opposite ped image.')
        if dmris_main and dmris_opposite and (
                len(dmris_main) == len(dmris_opposite)):
            study_specs.extend(
                [SubStudySpec('dwi_{}_main'.format(i),
                              CoregisteredDiffusionStudy,
                              ref_spec) for i in range(len(dmris_main))])
            inputs.update({'dwi_{}_main_dwi_main'.format(i):
                           Dataset(dmris_main_scan[0], dicom_format) for i,
                           dmris_main_scan in enumerate(dmris_main)})
            inputs.update({'dwi_{}_main_dwi_main_ref'.format(i):
                           Dataset(dmris_opposite[i][0], dicom_format) for i
                           in range(len(dmris_main))})
            if not dmris_ref:
                study_specs.extend(
                    [SubStudySpec('dwi_{}_opposite'.format(i),
                                  CoregisteredDiffusionReferenceOppositeStudy, ref_spec)
                     for i in range(len(dmris_main))])
                inputs.update(
                    {'dwi_{}_opposite_opposite_dwi2ref_ref'.format(i):
                     Dataset(dmris_main_scan[0], dicom_format) for i,
                     dmris_main_scan in enumerate(dmris_main)})
                inputs.update(
                    {'dwi_{}_opposite_opposite_dwi2ref_to_correct'.format(i):
                     Dataset(dmris_opposite[i][0], dicom_format)
                     for i in range(len(dmris_main))})
        elif dmris_main and dmris_opposite and (
                len(dmris_main) != len(dmris_opposite)):
            study_specs.extend(
                [SubStudySpec('dwi_{}_main'.format(i),
                              CoregisteredDiffusionStudy,
                              ref_spec) for i in range(len(dmris_main))])
            inputs.update({'dwi_{}_main_dwi_main'.format(i):
                           Dataset(dmris_main_scan[0], dicom_format) for i,
                           dmris_main_scan in enumerate(dmris_main)})
            inputs.update({'dwi_{}_main_dwi_main_ref'.format(i):
                           Dataset(dmris_opposite[0][0], dicom_format) for i
                           in range(len(dmris_main))})
            if not dmris_ref:
                study_specs.extend(
                    [SubStudySpec('dwi_{}_opposite'.format(i),
                                  CoregisteredDiffusionReferenceOppositeStudy, ref_spec)
                     for i in range(len(dmris_opposite))])
                inputs.update(
                    {'dwi_{}_opposite_opposite_dwi2ref_to_correct'.format(i):
                     Dataset(dmris_opp_scan[0], dicom_format) for i,
                     dmris_opp_scan in enumerate(dmris_opposite)})
                inputs.update(
                    {'dwi_{}_opposite_opposite_dwi2ref_ref'.format(i):
                     Dataset(dmris_main[0][0], dicom_format) for i
                     in range(len(dmris_opposite))})
        if dmris_ref and (len(dmris_ref) == len(dmris_opposite)):
            study_specs.extend([SubStudySpec('dwi_{}_toref'.format(i),
                                             CoregisteredDiffusionReferenceStudy, ref_spec)
                                for i in range(len(dmris_ref))])
            study_specs.extend(
                [SubStudySpec('dwi_{}_opposite'.format(i),
                              CoregisteredDiffusionReferenceOppositeStudy, ref_spec)
                 for i in range(len(dmris_ref))])
            inputs.update(
                {'dwi_{}_opposite_opposite_dwi2ref_to_correct'.format(i):
                 Dataset(dmris_opposite[i][0], dicom_format)
                 for i in range(len(dmris_ref))})
            inputs.update({'dwi_{}_opposite_opposite_dwi2ref_ref'.format(i):
                           Dataset(dmris_ref_scan[0], dicom_format) for i,
                           dmris_ref_scan in enumerate(dmris_ref)})
            inputs.update({'dwi_{}_toref_dwi2ref_ref'.format(i):
                           Dataset(dmris_opposite[i][0], dicom_format)
                           for i in range(len(dmris_ref))})
            inputs.update({'dwi_{}_toref_dwi2ref_to_correct'.format(i):
                           Dataset(dmris_ref_scan[0], dicom_format) for i,
                           dmris_ref_scan in enumerate(dmris_ref)})
        elif dmris_ref and (len(dmris_ref) != len(dmris_opposite)):
            study_specs.extend([SubStudySpec('dwi_{}_toref'.format(i),
                                             CoregisteredDiffusionReferenceStudy, ref_spec)
                                for i in range(len(dmris_ref))])
            inputs.update({'dwi_{}_toref_dwi2ref_ref'.format(i):
                           Dataset(dmris_opposite[0][0], dicom_format)
                           for i in range(len(dmris_ref))})
            inputs.update({'dwi_{}_toref_dwi2ref_to_correct'.format(i):
                           Dataset(dmris_ref_scan[0], dicom_format) for i,
                           dmris_ref_scan in enumerate(dmris_ref)})

            study_specs.extend(
                [SubStudySpec('dwi_{}_opposite'.format(i),
                              CoregisteredDiffusionReferenceOppositeStudy, ref_spec)
                 for i in range(len(dmris_opposite))])
            inputs.update({'dwi_{}_opposite_opposite_dwi2ref_ref'.format(i):
                           Dataset(dmris_ref[0][0], dicom_format)
                           for i in range(len(dmris_opposite))})
            inputs.update(
                {'dwi_{}_opposite_opposite_dwi2ref_to_correct'.format(i):
                 Dataset(dmris_opp_scan[0], dicom_format) for i,
                 dmris_ref_scan in enumerate(dmris_opposite)})
        run_pipeline = True

    if not run_pipeline:
        raise Exception('At least one scan, other than the reference, must be '
                        'provided!')
    dct['sub_study_specs'] = (
        MotionCorrectionMixin.sub_study_specs + study_specs)
    dct['add_data_specs'] = data_specs
    return (MultiStudyMetaClass(name, (MotionCorrectionMixin,), dct),
            inputs)
