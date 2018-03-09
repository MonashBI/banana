from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.data_formats import (
    nifti_gz_format, text_matrix_format, directory_format, text_format,
    png_format, dicom_format)
from mbianalysis.interfaces.custom.motion_correction import (
    MeanDisplacementCalculation, MotionFraming, PlotMeanDisplacementRC,
    AffineMatAveraging, PetCorrectionFactor, FrameAlign2Reference)
from nianalysis.citations import fsl_cite
from nianalysis.study.base import set_specs
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from .epi import CoregisteredEPIStudy
from .structural.t1 import CoregisteredT1Study, T1Study
from .structural.t2 import CoregisteredT2Study, T2Study
from nipype.interfaces.utility import Merge
from .structural.diffusion_coreg import (
    CoregisteredDiffusionStudy, CoregisteredDiffusionReferenceOppositeStudy,
    CoregisteredDiffusionReferenceStudy)
from nianalysis.requirements import fsl509_req
from nianalysis.exceptions import NiAnalysisNameError
from nianalysis.dataset import Dataset


class MotionReferenceT1Study(T1Study):

    def header_info_extraction_pipeline(self, reference=True, multivol=False,
                                        **kwargs):
        return (super(MotionReferenceT1Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', ref=reference, multivol=multivol,
                    **kwargs))

    def segmentation_pipeline(self, img_type=1, **kwargs):
        pipeline = super(MotionReferenceT1Study, self).segmentation_pipeline(
            img_type=img_type, **kwargs)
        return pipeline

    _data_specs = set_specs(
        DatasetSpec('wm_seg', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('motion_mats', directory_format,
                    header_info_extraction_pipeline),
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
        DatasetSpec('dcm_info', text_format, header_info_extraction_pipeline),
        inherit_from=T1Study.data_specs())


class MotionReferenceT2Study(T2Study):

    def header_info_extraction_pipeline(self, reference=True, multivol=False,
                                        **kwargs):
        return (super(MotionReferenceT1Study, self).
                header_info_extraction_pipeline_factory(
                    'primary', ref=reference, multivol=multivol,
                    **kwargs))

    def segmentation_pipeline(self, img_type=2, **kwargs):
        pipeline = super(MotionReferenceT1Study, self).segmentation_pipeline(
            img_type=img_type, **kwargs)
        return pipeline

    _data_specs = set_specs(
        DatasetSpec('wm_seg', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('motion_mats', directory_format,
                    header_info_extraction_pipeline),
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
        DatasetSpec('dcm_info', text_format, header_info_extraction_pipeline),
        inherit_from=T1Study.data_specs())


class MotionDetectionMixin(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    def mean_displacement_pipeline(self, **options):
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
                     DatasetSpec('offset_indexes', text_format),
                     DatasetSpec('mats4average', text_format)],
            description=("Calculate the mean displacement between each motion"
                         " matrix and a reference."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

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
        pipeline.connect_output('motion_par_rc', md, 'motion_parameters')
        pipeline.connect_output('offset_indexes', md, 'offset_indexes')
        pipeline.connect_output('mats4average', md, 'mats4average')
        pipeline.assert_connected()
        return pipeline

    def motion_framing_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='motion_framing',
            inputs=[DatasetSpec('mean_displacement', text_format),
                    DatasetSpec('mean_displacement_consecutive', text_format),
                    DatasetSpec('start_times', text_format)],
            outputs=[DatasetSpec('frame_start_times', text_format),
                     DatasetSpec('frame_vol_numbers', text_format)],
            description=("Calculate when the head movement exceeded a "
                         "predefined threshold (default 2mm)."),
            default_options={'th': 2.0, 'temporal_th': 30.0},
            version=1,
            citations=[fsl_cite],
            options=options)

        framing = pipeline.create_node(MotionFraming(), name='motion_framing')
        framing.inputs.motion_threshold = pipeline.option('th')
        framing.inputs.temporal_threshold = pipeline.option('temporal_th')
        pipeline.connect_input('mean_displacement', framing,
                               'mean_displacement')
        pipeline.connect_input('mean_displacement_consecutive', framing,
                               'mean_displacement_consec')
        pipeline.connect_input('start_times', framing, 'start_times')
        pipeline.connect_output('frame_start_times', framing,
                                'frame_start_times')
        pipeline.connect_output('frame_vol_numbers', framing,
                                'frame_vol_numbers')
        pipeline.assert_connected()
        return pipeline

    def plot_mean_displacement_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='plot_mean_displacement',
            inputs=[DatasetSpec('mean_displacement_rc', text_format),
                    DatasetSpec('offset_indexes', text_format),
                    DatasetSpec('frame_start_times', text_format)],
            outputs=[DatasetSpec('mean_displacement_plot', png_format)],
            description=("Plot the mean displacement real clock"),
            default_options={'framing': True},
            version=1,
            citations=[fsl_cite],
            options=options)

        plot_md = pipeline.create_node(PlotMeanDisplacementRC(),
                                       name='plot_md')
        plot_md.inputs.framing = pipeline.option('framing')
        pipeline.connect_input('mean_displacement_rc', plot_md,
                               'mean_disp_rc')
        pipeline.connect_input('offset_indexes', plot_md,
                               'false_indexes')
        pipeline.connect_input('frame_start_times', plot_md,
                               'frame_start_times')
        pipeline.connect_output('mean_displacement_plot', plot_md,
                                'mean_disp_plot')
        pipeline.assert_connected()
        return pipeline

    def frame_mean_transformation_mats_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='frame_mean_transformation_mats',
            inputs=[DatasetSpec('mats4average', text_format),
                    DatasetSpec('frame_vol_numbers', text_format)],
            outputs=[DatasetSpec('average_mats', directory_format)],
            description=("Average all the transformation mats within each "
                         "detected frame."),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        average = pipeline.create_node(AffineMatAveraging(),
                                       name='mats_averaging')
        pipeline.connect_input('frame_vol_numbers', average,
                               'frame_vol_numbers')
        pipeline.connect_input('mats4average', average,
                               'all_mats4average')
        pipeline.connect_output('average_mats', average,
                                'average_mats')
        pipeline.assert_connected()
        return pipeline

    def pet_correction_factors_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='pet_correction_factors',
            inputs=[DatasetSpec('frame_start_times', text_format)],
            outputs=[DatasetSpec('correction_factors', text_format)],
            description=("Pipeline to calculate the correction factors to "
                         "account for frame duration when averaging the PET "
                         "frames to create the static PET image"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        corr_factors = pipeline.create_node(PetCorrectionFactor(),
                                            name='pet_corr_factors')
        pipeline.connect_input('frame_start_times', corr_factors,
                               'frame_start_times')
        pipeline.connect_output('correction_factors', corr_factors,
                                'corr_factors')
        pipeline.assert_connected()
        return pipeline

    def frame2ref_alignment_pipeline_factory(
            self, name, average_mats, ute_regmat, ute_qform_mat, umap=None,
            pct=False, fixed_binning=False, **options):
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
            default_options={'pct': pct, 'fixed_binning': fixed_binning},
            version=1,
            citations=[fsl_cite],
            options=options)

        frame_align = pipeline.create_node(
            FrameAlign2Reference(), name='frame2ref_alignment',
            requirements=[fsl509_req])
        frame_align.inputs.pct = pipeline.option('pct')
        frame_align.inputs.fixed_binning = pipeline.option('fixed_binning')
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
        pipeline.assert_connected()
        return pipeline

    def frame2ref_alignment_pipeline(self, **options):
        return self.frame2ref_alignment_pipeline_factory(
            'frame2ref_alignment', 'average_mats', 'ute_reg_mat',
            'ute_qform_mat', umap='umap',
            pct=False, fixed_binning=False, **options)

  
def create_motion_detection_class(name, reference, ref_type, t1s=None,
                                  t2s=None, dmris=None, epis=None):

    inputs = {}
    if ref_type == 't1':
        ref_study = MotionReferenceT1Study
    elif ref_type == 't2':
        ref_study = MotionReferenceT2Study

    study_specs = [SubStudySpec('ref', ref_study)]
    ref_spec = {'ref_preproc': 'ref_preproc',
                'ref_masked': 'ref_brain',
                'ref_brain_mask': 'ref_brain_mask'}
    inputs['ref_primary'] = Dataset(reference, dicom_format)
    
    if t1s is not None:
        for i, t1_scan in enumerate(t1s):
            study_specs.extend(
                SubStudySpec('t1_{}_t1'.format(i), CoregisteredT1Study,
                             ref_spec))
            inputs['t1_{}_t1'.format(i)] = Dataset(t1_scan, dicom_format)
    if t2s is not None:
        for i, t2_scan in enumerate(t2s):
            study_specs.extend(
                SubStudySpec('t2_{}_t2'.format(i), CoregisteredT2Study,
                             ref_spec))
            inputs['t2_{}_t2'.format(i)] = Dataset(t2_scan, dicom_format)
    if epis is not None:
        for i, epi_scan in enumerate(epis):
            study_specs.extend(
                SubStudySpec('epi_{}_epi'.format(i), CoregisteredEPIStudy,
                             ref_spec.update({'ref_wm_seg': 'ref_wmseg'})))
            inputs['epi_{}_epi'.format(i)] = Dataset(epi_scan, dicom_format)
    if dmris is not None:
        for i, dmri_scan in enumerate(dmris):
            if dmri_scan[-1] == '0':
                study_specs.extend(
                    SubStudySpec('t2_{}_t2'.format(i), CoregisteredT2Study,
                                 ref_spec))
            inputs['t2_{}_t2'.format(i)] = Dataset(t2_scan, dicom_format)
    dct = {}
    dct['_sub_study_specs'] = set_specs(study_specs)
    dct['_data_specs'] = {}
    return MultiStudyMetaClass(name, [MotionDetectionMixin], dct), inputs


# class MotionReferenceMixin(MRIStudy):
#     def header_info_extraction_pipeline(self, reference=True, multivol=False,
#                                         **kwargs):
#         return (super(MotionReferenceStudy, self).
#                 header_info_extraction_pipeline_factory(
#                     'primary', ref=reference, multivol=multivol,
#                     **kwargs))
#     _data_specs = set_specs(
#         DatasetSpec('ref_motion_mats', directory_format,
#                     header_info_extraction_pipeline),
#         inherit_from=MRIStudy.data_specs())
