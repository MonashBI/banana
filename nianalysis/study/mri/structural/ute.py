from ..base import MRIStudy
from nianalysis.study.base import set_dataset_specs
from nianalysis.dataset import DatasetSpec
from nipype.interfaces.fsl.preprocess import FLIRT, ApplyXFM
from nipype.interfaces.fsl.utils import ConvertXFM
from nipype.interfaces.fsl.maths import UnaryMaths, BinaryMaths, MultiImageMaths, Threshold, SpatialFilter
from nipype.interfaces.spm.preprocess import NewSegment
from nipype.interfaces.utility.base import Select

from nianalysis.citations import (
    fsl_cite, spm_cite, matlab_cite)
from nianalysis.data_formats import (
    dicom_format, nifti_gz_format, nifti_format, text_matrix_format)
from nianalysis.requirements import (
    fsl5_req, mrtrix3_req, spm12_req, matlab2015_req)
from findertools import select

class UTEStudy(MRIStudy):
    
    #template_path = '/home/jakubb/template/template_template0.nii.gz'
    template_path = '/Users/jakubb/Desktop/ACProject/template/template_template0.nii.gz' 
    #tpm_path = '/environment/packages/spm/12/tpm/head_tpm.nii'
    tpm_path = '/Users/jakubb/Desktop/ACProject/template/head_tpm.nii'

    def registration_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Register T1 and T2 to the 

        Parameters
        ----------
        """
        pipeline = self.create_pipeline(
            name='registration_pipeline',
            inputs=[DatasetSpec('ute_echo1', nifti_gz_format),
                    DatasetSpec('ute_echo2', nifti_gz_format)],
            outputs=[DatasetSpec('ute1_registered', nifti_format),
                     DatasetSpec('ute2_registered', nifti_gz_format),
                     DatasetSpec('template_to_ute_mat', text_matrix_format),
                     DatasetSpec('ute_to_template_mat', text_matrix_format)],
            description="Register ute images to the template",
            default_options={},
            version=1,
            citations=(fsl_cite),
            options=options)
        
        # Create registration node
        registration = pipeline.create_node(
            FLIRT(), name='ute1_registration',
            requirements=[fsl5_req], wall_time=180)
        
        registration.inputs.reference = self.template_path 
        registration.inputs.output_type = 'NIFTI_GZ'
        registration.inputs.searchr_x = [-180, 180]
        registration.inputs.searchr_y = [-180, 180]
        registration.inputs.searchr_z = [-180, 180]
        registration.inputs.bins = 256
        registration.inputs.cost_func = 'corratio'
        pipeline.connect_input('ute_echo1', registration, 'in_file')
        
        # Inverse matrix conversion 
        convert_mat = pipeline.create_node(
            ConvertXFM(), name='inverse_matrix_conversion',
            requirements=[fsl5_req], wall_time=10)
        
        convert_mat.inputs.invert_xfm = True
        pipeline.connect(registration, 'out_matrix_file',  convert_mat, 'in_file')
        
        # UTE_echo_2 transformation
        transform_ute2 = pipeline.create_node(
            ApplyXFM(), name='transform_t2',
            requirements=[fsl5_req], wall_time=10)
        
        transform_ute2.inputs.output_type = 'NIFTI_GZ'
        transform_ute2.inputs.reference = self.template_path
        transform_ute2.inputs.apply_xfm = True
        pipeline.connect(registration, 'out_matrix_file',  transform_ute2, 'in_matrix_file')
        pipeline.connect_input('ute_echo2',  transform_ute2,'in_file')
        
        pipeline.connect_output('ute1_registered', registration, 'out_file')
        pipeline.connect_output('ute_to_template_mat',  registration, 'out_matrix_file')
        pipeline.connect_output('ute2_registered', transform_ute2, 'out_file')
        pipeline.connect_output('template_to_ute_mat',  convert_mat, 'out_file')
        pipeline.assert_connected()
        
        return pipeline


    def segmentation_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        

        pipeline = self.create_pipeline(
            name='ute1_segmentation',
            inputs=[DatasetSpec('ute1_registered', nifti_format)],
            outputs=[DatasetSpec('air_mask', nifti_gz_format),
                     DatasetSpec('bones_mask', nifti_gz_format),],
            description="Segmentation of the first echo UTE image",
            default_options={},
            version=1,
            citations=(spm_cite, matlab_cite),
            options=options)
        
        segmentation = pipeline.create_node(
            NewSegment(), name='ute1_registered_segmentation',
            requirements=[matlab2015_req, spm12_req], wall_time=480)
        
        segmentation.inputs.affine_regularization = 'none'
        tissue1 = ((self.tpm_path, 1), 1, (True,False), (False, False))
        tissue2 = ((self.tpm_path, 2), 1, (True,False), (False, False))
        tissue3 = ((self.tpm_path, 3), 2, (True,False), (False, False))
        tissue4 = ((self.tpm_path, 4), 3, (True,False), (False, False))
        tissue5 = ((self.tpm_path, 5), 4, (True,False), (False, False))
        tissue6 = ((self.tpm_path, 6), 3, (True,False), (False, False))
        segmentation.inputs.tissues = [tissue1, tissue2, tissue3, tissue4, tissue5, tissue6]
        pipeline.connect_input('ute1_registered',  segmentation, 'channel_files')
        
        select_bones_pm = pipeline.create_node(
            Select(), name='select_bones_pm_from_SPM_new_segmentation',
            requirements=[], wall_time=5)
        
        select_bones_pm.inputs.index=3
        pipeline.connect(segmentation, 'native_class_images', select_bones_pm, 'inlist')
        
        select_air_pm = pipeline.create_node(
            Select(), name='select_air_pm_from_SPM_new_segmentation',
            requirements=[], wall_time=5)
        
        select_air_pm.inputs.index=5
        pipeline.connect(segmentation, 'native_class_images', select_air_pm, 'inlist')
        
        threshold_bones = pipeline.create_node(
            Threshold(), name='bones_probabilistic_map_thresholding',
            requirements=[fsl5_req], wall_time=5)
        threshold_bones.inputs.output_type = "NIFTI_GZ"
        threshold_bones.inputs.direction = 'below'
        threshold_bones.inputs.thresh = 0.2
        pipeline.connect(select_bones_pm, 'out', threshold_bones, 'in_file')
        
        binarize_bones = pipeline.create_node(
            UnaryMaths(), name='bones_probabilistic_map_binarization',
            requirements=[fsl5_req], wall_time=5)
        binarize_bones.inputs.output_type = "NIFTI_GZ"
        binarize_bones.inputs.operation = 'bin'
        pipeline.connect(threshold_bones, 'out_file',  binarize_bones, 'in_file')
        
        
        threshold_air = pipeline.create_node(
            Threshold(), name='air_probabilistic_maps_thresholding',
            requirements=[fsl5_req], wall_time=5)
        threshold_air.inputs.output_type = "NIFTI_GZ"
        threshold_air.inputs.direction = 'below'
        threshold_air.inputs.thresh = 0.1
        pipeline.connect(select_air_pm, 'out', threshold_air, 'in_file')
        
        binarize_air = pipeline.create_node(
            UnaryMaths(), name='air_probabilistic_map_binarization',
            requirements=[fsl5_req], wall_time=5)
        binarize_air.inputs.output_type = "NIFTI_GZ"
        binarize_air.inputs.operation = 'bin'
        pipeline.connect(threshold_air, 'out_file',  binarize_air, 'in_file')

              
        pipeline.connect_output('bones_mask', binarize_bones, 'out_file')
        pipeline.connect_output('air_mask', binarize_air, 'out_file')                
        pipeline.assert_connected()
        
        return pipeline
    # The list of study dataset_specs that are either primary from the scanner
    # (i.e. without a specified pipeline) or generated by processing pipelines
    _dataset_specs = set_dataset_specs(
        DatasetSpec('ute_echo1', dicom_format),
        DatasetSpec('ute_echo2', dicom_format),
        DatasetSpec('umap_ute', dicom_format),
        DatasetSpec('ute1_registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('ute2_registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('template_to_ute_mat', text_matrix_format, registration_pipeline),
        DatasetSpec('ute_to_template_mat', text_matrix_format, registration_pipeline),
        DatasetSpec('air_mask', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('bones_mask', nifti_gz_format, segmentation_pipeline),
        
        inherit_from=MRIStudy.generated_dataset_specs())
