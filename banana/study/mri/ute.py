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
    dicom_format, nifti_gz_format, text_matrix_format)
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
        
#             inputs=[FilesetSpec('ute_echo1', dicom_format),
#                     FilesetSpec('ute_echo2', dicom_format)],
#             outputs=[FilesetSpec('ute1_registered', nifti_format),
#                      FilesetSpec('ute2_registered', nifti_gz_format),
#                      FilesetSpec('template_to_ute_mat', text_matrix_format),
#                      FilesetSpec('ute_to_template_mat', text_matrix_format)],
        
        pipeline = self.new_pipeline(
            name='registration_pipeline',
            desc="Register ute images to the template",
            citations=(fsl_cite),
            **kwargs)

        echo1_conv = pipeline.add(
            'echo1_conv',
            MRConvert())
        echo1_conv.inputs.out_ext = '.nii.gz'

        pipeline.connect_input('ute_echo1', echo1_conv, 'in_file')

        echo2_conv = pipeline.add(
            'echo2_conv',
            MRConvert())
        echo2_conv.inputs.out_ext = '.nii.gz'

        pipeline.connect_input('ute_echo2', echo2_conv, 'in_file')

        # Create registration node
        registration = pipeline.add(
            'ute1_registration',
            FLIRT(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=180)

        pipeline.connect(
            echo1_conv,
            'out_file',
            registration,
            'in_file')

        registration.inputs.reference = self.template_path
        registration.inputs.output_type = 'NIFTI_GZ'
        registration.inputs.searchr_x = [-180, 180]
        registration.inputs.searchr_y = [-180, 180]
        registration.inputs.searchr_z = [-180, 180]
        registration.inputs.bins = 256
        registration.inputs.cost_func = 'corratio'

        # Inverse matrix conversion
        convert_mat = pipeline.add(
            'inverse_matrix_conversion',
            ConvertXFM(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=10)
        pipeline.connect(
            registration,
            'out_matrix_file',
            convert_mat,
            'in_file')
        convert_mat.inputs.invert_xfm = True

        # UTE_echo_2 transformation
        transform_ute2 = pipeline.add(
            'transform_t2',
            ApplyXFM(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=10)
        pipeline.connect(
            registration,
            'out_matrix_file',
            transform_ute2,
            'in_matrix_file')
        pipeline.connect(
            echo2_conv,
            'out_file',
            transform_ute2,
            'in_file')

        transform_ute2.inputs.output_type = 'NIFTI_GZ'
        transform_ute2.inputs.reference = self.template_path
        transform_ute2.inputs.apply_xfm = True

        # Connect outputs
        pipeline.connect_output('ute1_registered', registration, 'out_file')
        pipeline.connect_output(
            'ute_to_template_mat',
            registration,
            'out_matrix_file')
        pipeline.connect_output('ute2_registered', transform_ute2, 'out_file')
        pipeline.connect_output('template_to_ute_mat', convert_mat, 'out_file')

        return pipeline

    def segmentation_pipeline(self, **kwargs):  # @UnusedVariable @IgnorePep8


#             inputs=[FilesetSpec('ute1_registered', nifti_format)],
#             outputs=[FilesetSpec('air_mask', nifti_gz_format),
#                      FilesetSpec('bones_mask', nifti_gz_format)],

        pipeline = self.new_pipeline(
            name='ute1_segmentation',
            desc="Segmentation of the first echo UTE image",
            citations=(spm_cite, matlab_cite),
            **kwargs)

        segmentation = pipeline.add(
            'ute1_registered_segmentation',
            NewSegment(),
            requirements=[matlab_req.v('R2015'), spm_req.v('12')],
            wall_time=480)
        pipeline.connect_input(
            'ute1_registered',
            segmentation,
            'channel_files')
        segmentation.inputs.affine_regularization = 'none'
        tissue1 = ((self.tpm_path, 1), 1, (True, False), (False, False))
        tissue2 = ((self.tpm_path, 2), 1, (True, False), (False, False))
        tissue3 = ((self.tpm_path, 3), 2, (True, False), (False, False))
        tissue4 = ((self.tpm_path, 4), 3, (True, False), (False, False))
        tissue5 = ((self.tpm_path, 5), 4, (True, False), (False, False))
        tissue6 = ((self.tpm_path, 6), 3, (True, False), (False, False))
        segmentation.inputs.tissues = [
            tissue1,
            tissue2,
            tissue3,
            tissue4,
            tissue5,
            tissue6]

        select_bones_pm = pipeline.add(
            'select_bones_pm_from_SPM_new_segmentation',
            Select(),
            requirements=[],
            wall_time=5)
        pipeline.connect(
            segmentation,
            'native_class_images',
            select_bones_pm,
            'inlist')
        select_bones_pm.inputs.index = 3

        select_air_pm = pipeline.add(
            'select_air_pm_from_SPM_new_segmentation',
            Select(),
            requirements=[],
            wall_time=5)

        pipeline.connect(
            segmentation,
            'native_class_images',
            select_air_pm,
            'inlist')
        select_air_pm.inputs.index = 5

        threshold_bones = pipeline.add(
            'bones_probabilistic_map_thresholding',
            Threshold(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(select_bones_pm, 'out', threshold_bones, 'in_file')
        threshold_bones.inputs.output_type = "NIFTI_GZ"
        threshold_bones.inputs.direction = 'below'
        threshold_bones.inputs.thresh = 0.2

        binarize_bones = pipeline.add(
            'bones_probabilistic_map_binarization',
            UnaryMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(
            threshold_bones,
            'out_file',
            binarize_bones,
            'in_file')
        binarize_bones.inputs.output_type = "NIFTI_GZ"
        binarize_bones.inputs.operation = 'bin'

        threshold_air = pipeline.add(
            'air_probabilistic_maps_thresholding',
            Threshold(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(select_air_pm, 'out', threshold_air, 'in_file')
        threshold_air.inputs.output_type = "NIFTI_GZ"
        threshold_air.inputs.direction = 'below'
        threshold_air.inputs.thresh = 0.1

        binarize_air = pipeline.add(
            'air_probabilistic_map_binarization',
            UnaryMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(threshold_air, 'out_file', binarize_air, 'in_file')
        binarize_air.inputs.output_type = "NIFTI_GZ"
        binarize_air.inputs.operation = 'bin'

        pipeline.connect_output('bones_mask', binarize_bones, 'out_file')
        pipeline.connect_output('air_mask', binarize_air, 'out_file')

        return pipeline

    def umaps_calculation_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('ute1_registered', nifti_gz_format),
#                     FilesetSpec('ute2_registered', nifti_gz_format),
#                     FilesetSpec('air_mask', nifti_gz_format),
#                     FilesetSpec('bones_mask', nifti_gz_format)],
#             outputs=[FilesetSpec('sute_cont_template', nifti_gz_format),
#                      FilesetSpec('sute_fix_template', nifti_gz_format)],


        pipeline = self.new_pipeline(
            name='core_umaps_calculation',
            desc="Umaps calculation in the template space",
            citations=(matlab_cite),
            **kwargs)

        umaps_calculation = pipeline.add(
            'umaps_calculation_based_on_masks_and_r2star',
            CoreUmapCalc(),
            requirements=[matlab_req.v('R2015')],
            wall_time=20)
        pipeline.connect_input(
            'ute1_registered',
            umaps_calculation,
            'ute1_reg')
        pipeline.connect_input(
            'ute2_registered',
            umaps_calculation,
            'ute2_reg')
        pipeline.connect_input('air_mask', umaps_calculation, 'air__mask')
        pipeline.connect_input('bones_mask', umaps_calculation, 'bones__mask')

        pipeline.connect_output(
            'sute_cont_template',
            umaps_calculation,
            'sute_cont_template')
        pipeline.connect_output(
            'sute_fix_template',
            umaps_calculation,
            'sute_fix_template')

        return pipeline

    def backwrap_to_ute_pipeline(self, **kwargs):

#             inputs=[FilesetSpec('ute1_registered', nifti_gz_format),
#                     FilesetSpec('ute_echo1', dicom_format),
#                     FilesetSpec('umap_ute', dicom_format),
#                     FilesetSpec('template_to_ute_mat', text_matrix_format),
#                     FilesetSpec('sute_cont_template', nifti_gz_format),
#                     FilesetSpec('sute_fix_template', nifti_gz_format)],
#             outputs=[FilesetSpec('sute_cont_ute', nifti_gz_format),
#                      FilesetSpec('sute_fix_ute', nifti_gz_format)],


        pipeline = self.new_pipeline(
            name='backwrap_to_ute',
            desc="Moving umaps back to the UTE space",
            citations=(matlab_cite),
            **kwargs)

        echo1_conv = pipeline.add(
            'echo1_conv',
            MRConvert())
        echo1_conv.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('ute_echo1', echo1_conv, 'in_file')

        umap_conv = pipeline.add(
            'umap_conv',
            MRConvert())
        umap_conv.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('umap_ute', umap_conv, 'in_file')

        zero_template_mask = pipeline.add(
            'zero_template_mask',
            BinaryMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=3)
        pipeline.connect_input(
            'ute1_registered',
            zero_template_mask,
            'in_file')
        zero_template_mask.inputs.operation = "mul"
        zero_template_mask.inputs.operand_value = 0
        zero_template_mask.inputs.output_type = 'NIFTI_GZ'

        region_template_mask = pipeline.add(
            'region_template_mask',
            FLIRT(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        region_template_mask.inputs.apply_xfm = True
        region_template_mask.inputs.bgvalue = 1
        region_template_mask.inputs.interp = 'nearestneighbour'
        region_template_mask.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(
            zero_template_mask,
            'out_file',
            region_template_mask,
            'in_file')
        pipeline.connect(
            echo1_conv,
            'out_file',
            region_template_mask,
            'reference')
        pipeline.connect_input('template_to_ute_mat', region_template_mask,
                               'in_matrix_file')

        fill_in_umap = pipeline.add(
            'fill_in_umap',
            MultiImageMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=3)
        fill_in_umap.inputs.op_string = "-mul %s "
        fill_in_umap.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(region_template_mask, 'out_file',
                         fill_in_umap, 'in_file')
        pipeline.connect(
            umap_conv,
            'out_file',
            fill_in_umap,
            'operand_files')

        sute_fix_ute_space = pipeline.add(
            'sute_fix_ute_space',
            FLIRT(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(
            echo1_conv,
            'out_file',
            sute_fix_ute_space,
            'reference')
        pipeline.connect_input('template_to_ute_mat', sute_fix_ute_space,
                               'in_matrix_file')
        pipeline.connect_input('sute_fix_template', sute_fix_ute_space,
                               'in_file')
        sute_fix_ute_space.inputs.apply_xfm = True
        sute_fix_ute_space.inputs.bgvalue = 0
        sute_fix_ute_space.inputs.output_type = 'NIFTI_GZ'

        sute_cont_ute_space = pipeline.add(
            'sute_cont_ute_space',
            FLIRT(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(
            echo1_conv,
            'out_file',
            sute_cont_ute_space,
            'reference')
        pipeline.connect_input('template_to_ute_mat', sute_cont_ute_space,
                               'in_matrix_file')
        pipeline.connect_input('sute_cont_template', sute_cont_ute_space,
                               'in_file')
        sute_cont_ute_space.inputs.apply_xfm = True
        sute_cont_ute_space.inputs.bgvalue = 0
        sute_cont_ute_space.inputs.output_type = 'NIFTI_GZ'

        sute_fix_ute_background = pipeline.add(
            'sute_fix_ute_background',
            MultiImageMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(
            sute_fix_ute_space,
            'out_file',
            sute_fix_ute_background,
            'in_file')
        sute_fix_ute_background.inputs.op_string = "-add %s "
        sute_fix_ute_background.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(
            fill_in_umap,
            'out_file',
            sute_fix_ute_background,
            'operand_files')

        sute_cont_ute_background = pipeline.add(
            'sute_cont_ute_background',
            MultiImageMaths(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        pipeline.connect(
            sute_cont_ute_space,
            'out_file',
            sute_cont_ute_background,
            'in_file')
        sute_cont_ute_background.inputs.op_string = "-add %s "
        sute_cont_ute_background.inputs.output_type = 'NIFTI_GZ'
        pipeline.connect(
            fill_in_umap,
            'out_file',
            sute_cont_ute_background,
            'operand_files')

        smooth_sute_fix = pipeline.add(
            'smooth_sute_fix',
            Smooth(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        smooth_sute_fix.inputs.sigma = 2.
        pipeline.connect(
            sute_fix_ute_background,
            'out_file',
            smooth_sute_fix,
            'in_file')

        smooth_sute_cont = pipeline.add(
            'smooth_sute_cont',
            Smooth(),
            requirements=[fsl_req.v('5.0.10')],
            wall_time=5)
        smooth_sute_cont.inputs.sigma = 2.
        pipeline.connect(
            sute_cont_ute_background,
            'out_file',
            smooth_sute_cont,
            'in_file')

        pipeline.connect_output('sute_fix_ute', smooth_sute_fix,
                                'smoothed_file')
        pipeline.connect_output('sute_cont_ute', smooth_sute_cont,
                                'smoothed_file')

        return pipeline

#     def conversion_to_dicom_pipeline(self, **kwargs):
#
#         pipeline = self.new_pipeline(
#             name='conversion_to_dicom',
#             inputs=[FilesetSpec('sute_cont_ute', nifti_gz_format),
#                     FilesetSpec('sute_fix_ute', nifti_gz_format),
#                     FilesetSpec('umap_ute', dicom_format)],
#             outputs=[FilesetSpec('sute_cont_dicoms', dicom_format),
#                      FilesetSpec('sute_fix_dicoms', dicom_format)],
#             desc=(
#                 "Conversing resulted two umaps from nifti to dicom format - "
#                 "parallel implementation"),
#             version=1,
#             citations=(),
#             parameters=parameters)
#
#         cont_split = pipeline.create_node(Split(), name='cont_split',
#                                           requirements=[fsl_req.v('5.0.10')])
#         cont_split.inputs.dimension = 'z'
#         fix_split = pipeline.create_node(Split(), name='fix_split',
#                                          requirements=[fsl_req.v('5.0.10')])
#         fix_split.inputs.dimension = 'z'
#         cont_nii2dicom = pipeline.create_map_node(
#             Nii2Dicom(), name='cont_nii2dicom', iterfield=['in_file',
#                                                            'reference_dicom'],
#             wall_time=20)
#         fix_nii2dicom = pipeline.create_map_node(
#             Nii2Dicom(), name='fix_nii2dicom', iterfield=['in_file',
#                                                           'reference_dicom'],
#             wall_time=20)
#         list_dicoms = pipeline.add('list_dicoms', ListDir())
#         list_dicoms.inputs.sort_key = dicom_fname_sort_key
#         cont_copy2dir = pipeline.create_node(CopyToDir(),
#                                              name='cont_copy2dir')
#         cont_copy2dir.inputs.file_ext = '.dcm'
#         fix_copy2dir = pipeline.create_node(CopyToDir(),
#                                             name='fix_copy2dir')
#         fix_copy2dir.inputs.file_ext = '.dcm'
#         # Connect nodes
#         pipeline.connect(cont_split, 'out_files', cont_nii2dicom, 'in_file')
#         pipeline.connect(fix_split, 'out_files', fix_nii2dicom, 'in_file')
#         pipeline.connect(list_dicoms, 'files', cont_nii2dicom,
#                          'reference_dicom')
#         pipeline.connect(list_dicoms, 'files', fix_nii2dicom,
#                          'reference_dicom')
#         pipeline.connect(cont_nii2dicom, 'out_file', cont_copy2dir,
#                          'in_files')
#         pipeline.connect(fix_nii2dicom, 'out_file', fix_copy2dir, 'in_files')
#         # Connect inputs
#         pipeline.connect_input('sute_cont_ute', cont_split, 'in_file')
#         pipeline.connect_input('sute_fix_ute', fix_split, 'in_file')
#         pipeline.connect_input('umap_ute', list_dicoms, 'directory')
#         # Connect outputs
#         pipeline.connect_output('sute_fix_dicoms', fix_copy2dir, 'out_dir')
#         pipeline.connect_output('sute_cont_dicoms', cont_copy2dir, 'out_dir')
#
#         pipeline.assert_connected()
#         return pipeline
    # The list of study data_specs that are either primary from the scanner
    # (i.e. without a specified pipeline) or generated by processing pipelines
#     add_data_specs = [
#         FilesetSpec(
#             'sute_fix_dicoms',
#             dicom_format,
#             conversion_to_dicom_pipeline),
#         FilesetSpec(
#             'sute_cont_dicoms',
#             dicom_format,
#             conversion_to_dicom_pipeline)]
