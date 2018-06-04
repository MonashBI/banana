from ..base import MRIStudy
from arcana.dataset import DatasetSpec, FieldSpec
from nianalysis.data_format import (
    nifti_gz_format, directory_format, dicom_format, eddy_par_format)
from nipype.interfaces.fsl import ExtractROI
from nianalysis.interfaces.custom.motion_correction import (
    PrepareDWI, CheckDwiNames)
from nianalysis.citation import fsl_cite
from arcana.study.base import StudyMetaClass
from nianalysis.interfaces.custom.motion_correction import (
    AffineMatrixGeneration)
from nipype.interfaces.utility import Merge as merge_lists
from nianalysis.interfaces.mrtrix.preproc import DWIPreproc
from nipype.interfaces.fsl.utils import Merge as fsl_merge
from nianalysis.requirement import fsl509_req, mrtrix3_req, fsl510_req
from nianalysis.interfaces.mrtrix import MRConvert
from arcana.option import OptionSpec
from nipype.interfaces import fsl


class DWIStudy(MRIStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        DatasetSpec('dwi_reference', dicom_format, optional=True),
        DatasetSpec('eddy_par', eddy_par_format, 'basic_preproc_pipeline'),
        DatasetSpec('align_mats', directory_format, 'affine_mats_pipeline')]

    add_option_specs = [
        OptionSpec('bet_robust', True),
        OptionSpec('bet_f_threshold', 0.2),
        OptionSpec('bet_reduce_bias', False)]

    def basic_preproc_pipeline(self, **kwargs):

        pipeline = self._eddy_dwipreproc_pipeline(**kwargs)
        return pipeline

    def _eddy_dwipreproc_pipeline(self, **kwargs):

        if 'dwi_reference' in self.input_names:
            inputs = [DatasetSpec('primary', dicom_format),
                      DatasetSpec('dwi_reference', dicom_format),
                      FieldSpec('ped', dtype=str),
                      FieldSpec('pe_angle', dtype=str)]
            distortion_correction = True
        else:
            inputs = [DatasetSpec('primary', dicom_format)]
            distortion_correction = False

        pipeline = self.create_pipeline(
            name='eddy_preproc',
            inputs=inputs,
            outputs=[DatasetSpec('preproc', nifti_gz_format),
                     DatasetSpec('eddy_par', eddy_par_format)],
            desc=("Diffusion pre-processing pipeline"),
            version=1,
            citations=[],
            **kwargs)

        if distortion_correction:
            converter1 = pipeline.create_node(MRConvert(), name='converter1',
                                              requirements=[mrtrix3_req])
            converter1.inputs.out_ext = '.nii.gz'
            pipeline.connect_input('primary', converter1, 'in_file')
            converter2 = pipeline.create_node(MRConvert(), name='converter2',
                                              requirements=[mrtrix3_req])
            converter2.inputs.out_ext = '.nii.gz'
            pipeline.connect_input('dwi_reference', converter2, 'in_file')
            prep_dwi = pipeline.create_node(PrepareDWI(), name='prepare_dwi')
            pipeline.connect_input('ped', prep_dwi, 'pe_dir')
            pipeline.connect_input('pe_angle', prep_dwi, 'phase_offset')
            pipeline.connect(converter1, 'out_file', prep_dwi, 'dwi')
            pipeline.connect(converter2, 'out_file', prep_dwi, 'dwi1')

            check_name = pipeline.create_node(CheckDwiNames(),
                                              name='check_names')
            pipeline.connect(prep_dwi, 'main', check_name, 'nifti_dwi')
            pipeline.connect_input('primary', check_name, 'dicom_dwi')
            pipeline.connect_input('dwi_reference', check_name, 'dicom_dwi1')
            roi = pipeline.create_node(ExtractROI(), name='extract_roi',
                                       requirements=[fsl509_req])
            roi.inputs.t_min = 0
            roi.inputs.t_size = 1
            pipeline.connect(prep_dwi, 'main', roi, 'in_file')

            merge_outputs = pipeline.create_node(merge_lists(2),
                                                 name='merge_files')
            pipeline.connect(roi, 'roi_file', merge_outputs, 'in1')
            pipeline.connect(prep_dwi, 'secondary', merge_outputs, 'in2')
            merge = pipeline.create_node(fsl_merge(), name='fsl_merge',
                                         requirements=[fsl509_req])
            merge.inputs.dimension = 't'
            pipeline.connect(merge_outputs, 'out', merge, 'in_files')

        dwipreproc = pipeline.create_node(
            DWIPreproc(), name='dwipreproc',
            requirements=[fsl510_req, mrtrix3_req])
        dwipreproc.inputs.eddy_options = '--data_is_shelled '
        dwipreproc.inputs.no_clean_up = True
        dwipreproc.inputs.out_file_ext = '.nii.gz'
        dwipreproc.inputs.temp_dir = 'dwipreproc_tempdir'
        if distortion_correction:
            dwipreproc.inputs.rpe_pair = True
            pipeline.connect(merge, 'merged_file', dwipreproc, 'se_epi')
            pipeline.connect(prep_dwi, 'pe', dwipreproc, 'pe_dir')
            pipeline.connect(check_name, 'main', dwipreproc, 'in_file')
        else:
            dwipreproc.inputs.rpe_header = True
            pipeline.connect_input('primary', dwipreproc, 'in_file')
        swap = pipeline.create_node(fsl.utils.Reorient2Std(),
                                    name='fslreorient2std',
                                    requirements=[fsl509_req])
        pipeline.connect(dwipreproc, 'out_file', swap, 'in_file')
        pipeline.connect_output('preproc', swap, 'out_file')
        pipeline.connect_output('eddy_par', dwipreproc, 'eddy_parameters')

        return pipeline

    def affine_mats_pipeline(self, **kwargs):

        pipeline = self.create_pipeline(
            name='affine_mat_generation',
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('eddy_par', eddy_par_format)],
            outputs=[
                DatasetSpec('align_mats', directory_format)],
            desc=("Generation of the affine matrices for the main dwi "
                  "sequence starting from eddy motion parameters"),
            version=1,
            citations=[fsl_cite],
            **kwargs)

        aff_mat = pipeline.create_node(AffineMatrixGeneration(),
                                       name='gen_aff_mats')
        pipeline.connect_input('preproc', aff_mat, 'reference_image')
        pipeline.connect_input(
            'eddy_par', aff_mat, 'motion_parameters')
        pipeline.connect_output(
            'align_mats', aff_mat, 'affine_matrices')
        return pipeline
