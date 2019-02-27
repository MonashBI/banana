from .base import MriStudy
from arcana.study.base import StudyMetaClass
from arcana.data import FilesetSpec
from nipype.interfaces.fsl.preprocess import FLIRT, ApplyXFM
from nipype.interfaces.fsl.utils import ConvertXFM, Smooth
from nipype.interfaces.fsl.maths import (
    UnaryMaths, BinaryMaths, MultiImageMaths, Threshold)
from nipype.interfaces.spm.preprocess import NewSegment
from nipype.interfaces.utility.base import Select
from banana.interfaces.umap_calc import CoreUmapCalc
from banana.interfaces.mrtrix.utils import MRConvert
from banana.citation import (
    fsl_cite, spm_cite, matlab_cite)
from banana.file_format import (
    dicom_format, nifti_gz_format, nifti_format, text_matrix_format)
from banana.requirement import (
    fsl_req, spm_req, matlab_req)
from arcana.study import SwitchSpec


class UteStudy(MriStudy, metaclass=StudyMetaClass):

    add_data_specs = [
        FilesetSpec('umap', dicom_format),
        FilesetSpec('umap_nifti', nifti_gz_format,
                    'umap_dcm2nii_conversion_pipeline'),
        FilesetSpec('brain', nifti_gz_format, 'brain_extraction_pipeline'),
        FilesetSpec('ute_echo1', dicom_format),
        FilesetSpec('ute_echo2', dicom_format),
        FilesetSpec('umap_ute', dicom_format),
        FilesetSpec('ute1_registered', nifti_gz_format,
                    'registration_pipeline'),
        FilesetSpec('ute2_registered', nifti_gz_format,
                    'registration_pipeline'),
        FilesetSpec('template_to_ute_mat', text_matrix_format,
                    'registration_pipeline'),
        FilesetSpec('ute_to_template_mat', text_matrix_format,
                    'registration_pipeline'),
        FilesetSpec('air_mask', nifti_gz_format,
                    'segmentation_pipeline'),
        FilesetSpec('bones_mask', nifti_gz_format,
                    'segmentation_pipeline'),
        FilesetSpec('sute_cont_template', nifti_gz_format,
                    'umaps_calculation_pipeline'),
        FilesetSpec('sute_fix_template', nifti_gz_format,
                    'umaps_calculation_pipeline'),
        FilesetSpec('sute_fix_ute', nifti_gz_format,
                    'backwrap_to_ute_pipeline'),
        FilesetSpec('sute_cont_ute', nifti_gz_format,
                    'backwrap_to_ute_pipeline')]

    add_param_specs = [
        SwitchSpec('bet_method', 'optibet',
                   choices=MriStudy.parameter_spec('bet_method').choices)]

    template_path = '/home/jakubb/template/template_template0.nii.gz'
    tpm_path = '/environment/packages/spm/12/tpm/head_tpm.nii'

    def header_extraction_pipeline(self, **kwargs):
        return (super(UteStudy, self).
                header_extraction_pipeline_factory(
                    'magnitude', **kwargs))

    def umap_dcm2nii_conversion_pipeline(self, **kwargs):
        return super(UteStudy, self).dcm2nii_conversion_pipeline_factory(
            'umap_dcm2nii', 'umap', **kwargs)

    def registration_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8
        """
        Register T1 and T2 to the

        Parameters
        ----------
        """

        pipeline = self.new_pipeline(
            name='registration_pipeline',
            desc="Register ute images to the template",
            citations=(fsl_cite),
            **kwargs)

        echo1_conv = pipeline.add(
            'echo1_conv',
            MRConvert(
                out_ext='.nii.gz'),
            inputs={
                'in_file': ('ute_echo1', dicom_format)},
            outputs={})

        echo2_conv = pipeline.add(
            'echo2_conv',
            MRConvert(
                out_ext='.nii.gz'),
            inputs={
                'in_file': ('ute_echo2', dicom_format)})

        # Create registration node
        registration = pipeline.add(
            'ute1_registration',
            FLIRT(
                reference=self.template_path,
                output_type='NIFTI_GZ',
                searchr_x=[-180, 180],
                searchr_y=[-180, 180],
                searchr_z=[-180, 180],
                bins=256,
                cost_func='corratio'),
            inputs={
                'in_file': (echo1_conv, 'out_file')},
            outputs={
                'ute1_registered': ('out_file', nifti_format),
                'ute_to_template_mat': ('out_matrix_file',
                                        text_matrix_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=180)

        # Inverse matrix conversion
        pipeline.add(
            'inverse_matrix_conversion',
            ConvertXFM(
                invert_xfm=True),
            inputs={
                'in_file': (registration, 'out_matrix_file')},
            outputs={
                'template_to_ute_mat': ('out_file', text_matrix_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=10)

        # UTE_echo_2 transformation
        pipeline.add(
            'transform_t2',
            ApplyXFM(
                output_type='NIFTI_GZ',
                reference=self.template_path,
                apply_xfm=True),
            inputs={
                'in_matrix_file': (registration, 'out_matrix_file'),
                'in_file': (echo2_conv, 'out_file')},
            outputs={
                'ute2_registered': ('out_file', nifti_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=10)

        # Connect outputs

        return pipeline

    def segmentation_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8

        pipeline = self.new_pipeline(
            name='ute1_segmentation',
            desc="Segmentation of the first echo UTE image",
            citations=(spm_cite, matlab_cite),
            **kwargs)

        segmentation = pipeline.add(
            'ute1_registered_segmentation',
            NewSegment(
                affine_regularization='none',
                tissues=[
                    ((self.tpm_path, 1), 1, (True, False), (False, False)),
                    ((self.tpm_path, 2), 1, (True, False), (False, False)),
                    ((self.tpm_path, 3), 2, (True, False), (False, False)),
                    ((self.tpm_path, 4), 3, (True, False), (False, False)),
                    ((self.tpm_path, 5), 4, (True, False), (False, False)),
                    ((self.tpm_path, 6), 3, (True, False), (False, False))]),
            inputs={
                'channel_files': ('ute1_registered', nifti_format)},
            requirements=[matlab_req.v('R2015'), spm_req.v('12')],
            wall_time=480)

        select_bones_pm = pipeline.add(
            'select_bones_pm_from_SPM_new_segmentation',
            Select(
                index=3),
            inputs={
                'inlist': (segmentation, 'native_class_images')},
            requirements=[],
            wall_time=5)

        select_air_pm = pipeline.add(
            'select_air_pm_from_SPM_new_segmentation',
            Select(
                index=5),
            inputs={
                'inlist': (segmentation, 'native_class_images')},
            requirements=[],
            wall_time=5)

        threshold_bones = pipeline.add(
            'bones_probabilistic_map_thresholding',
            Threshold(
                output_type="NIFTI_GZ",
                direction='below',
                thresh=0.2),
            inputs={
                'in_file': (select_bones_pm, 'out')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        pipeline.add(
            'bones_probabilistic_map_binarization',
            UnaryMaths(
                output_type="NIFTI_GZ",
                operation='bin'),
            inputs={
                'in_file': (threshold_bones, 'out_file')},
            outputs={
                'bones_mask': ('out_file', nifti_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        threshold_air = pipeline.add(
            'air_probabilistic_maps_thresholding',
            Threshold(
                output_type="NIFTI_GZ",
                direction='below',
                thresh=0.1),
            inputs={
                'in_file': (select_air_pm, 'out')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        pipeline.add(
            'air_probabilistic_map_binarization',
            UnaryMaths(
                output_type="NIFTI_GZ",
                operation='bin'),
            inputs={
                'in_file': (threshold_air, 'out_file')},
            outputs={
                'air_mask': ('out_file', nifti_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        return pipeline

    def umaps_calculation_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='core_umaps_calculation',
            desc="Umaps calculation in the template space",
            citations=(matlab_cite),
            **kwargs)

        pipeline.add(
            'umaps_calculation_based_on_masks_and_r2star',
            CoreUmapCalc(),
            inputs={
                'ute1_reg': ('ute1_registered', nifti_gz_format),
                'ute2_reg': ('ute2_registered', nifti_gz_format),
                'air__mask': ('air_mask', nifti_gz_format),
                'bones__mask': ('bones_mask', nifti_gz_format)},
            outputs={
                'sute_cont_template': ('sute_cont_template', nifti_gz_format),
                'sute_fix_template': ('sute_fix_template', nifti_gz_format)},
            requirements=[matlab_req.v('R2015')],
            wall_time=20)

        return pipeline

    def backwrap_to_ute_pipeline(self, **kwargs):

        pipeline = self.new_pipeline(
            name='backwrap_to_ute',
            desc="Moving umaps back to the UTE space",
            citations=(matlab_cite),
            **kwargs)

        echo1_conv = pipeline.add(
            'echo1_conv',
            MRConvert(
                out_ext='.nii.gz'),
            inputs={
                'in_file': ('ute_echo1', dicom_format)})

        umap_conv = pipeline.add(
            'umap_conv',
            MRConvert(
                out_ext='.nii.gz'),
            inputs={
                'in_file': ('umap_ute', dicom_format)})

        zero_template_mask = pipeline.add(
            'zero_template_mask',
            BinaryMaths(
                operation="mul",
                operand_value=0,
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': ('ute1_registered', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=3)

        region_template_mask = pipeline.add(
            'region_template_mask',
            FLIRT(
                apply_xfm=True,
                bgvalue=1,
                interp='nearestneighbour',
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (zero_template_mask, 'out_file'),
                'reference': (echo1_conv, 'out_file'),
                'in_matrix_file': ('template_to_ute_mat', text_matrix_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        fill_in_umap = pipeline.add(
            'fill_in_umap',
            MultiImageMaths(
                op_string="-mul %s ",
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (region_template_mask, 'out_file'),
                'operand_files': (umap_conv, 'out_file')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=3)

        sute_fix_ute_space = pipeline.add(
            'sute_fix_ute_space',
            FLIRT(
                apply_xfm=True,
                bgvalue=0,
                output_type='NIFTI_GZ'),
            inputs={
                'reference': (echo1_conv, 'out_file'),
                'in_matrix_file': ('template_to_ute_mat', nifti_gz_format),
                'in_file': ('sute_fix_template', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        sute_cont_ute_space = pipeline.add(
            'sute_cont_ute_space',
            FLIRT(
                apply_xfm=True,
                bgvalue=0,
                output_type='NIFTI_GZ'),
            inputs={
                'in_matrix_file': ('template_to_ute_mat', nifti_gz_format),
                'in_file': ('sute_cont_template', nifti_gz_format),
                'reference': (echo1_conv, 'out_file')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        sute_fix_ute_background = pipeline.add(
            'sute_fix_ute_background',
            MultiImageMaths(
                op_string="-add %s ",
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (sute_fix_ute_space, 'out_file'),
                'operand_files': (fill_in_umap, 'out_file')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        sute_cont_ute_background = pipeline.add(
            'sute_cont_ute_background',
            MultiImageMaths(
                op_string="-add %s ",
                output_type='NIFTI_GZ'),
            inputs={
                'in_file': (sute_cont_ute_space, 'out_file'),
                'operand_files': (fill_in_umap, 'out_file')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        pipeline.add(
            'smooth_sute_fix',
            Smooth(
                sigma=2.),
            inputs={
                'in_file': (sute_fix_ute_background, 'out_file')},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        pipeline.add(
            'smooth_sute_cont',
            Smooth(
                sigma=2.),
            inputs={
                'in_file': (sute_cont_ute_background, 'out_file')},
            outputs={
                'sute_fix_ute': ('smoothed_file', nifti_gz_format),
                'sute_cont_ute': ('smoothed_file', nifti_gz_format)},
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)

        return pipeline
