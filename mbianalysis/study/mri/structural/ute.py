from ..base import MRIStudy
from nianalysis.study.base import set_specs
from nianalysis.dataset import DatasetSpec, FieldSpec
from nipype.interfaces.fsl.preprocess import FLIRT, ApplyXFM
from nipype.interfaces.fsl.utils import ConvertXFM, Smooth
from nipype.interfaces.fsl.maths import (
    UnaryMaths, BinaryMaths, MultiImageMaths, Threshold)
from nipype.interfaces.spm.preprocess import NewSegment
from nipype.interfaces.utility.base import Select
from mbianalysis.interfaces.umap_calc import CoreUmapCalc
from nianalysis.interfaces.converters import Nii2Dicom
from mbianalysis.interfaces.mrtrix.utils import MRConvert
from nianalysis.interfaces.utils import (
    CopyToDir, ListDir, dicom_fname_sort_key)
from nianalysis.study.multi import (
    MultiStudy, SubStudySpec, MultiStudyMetaClass)
from ..coregistered import CoregisteredStudy
from nianalysis.citations import (
    fsl_cite, spm_cite, matlab_cite)
from nianalysis.data_formats import (
    dicom_format, nifti_gz_format, nifti_format, text_matrix_format,
    directory_format, text_format)
from nianalysis.requirements import (
    fsl5_req, spm12_req, matlab2015_req)
from mbianalysis.interfaces.custom.motion_correction import (
    MotionMatCalculation)


class UTEStudy(MRIStudy):

    template_path = '/home/jakubb/template/template_template0.nii.gz'
    tpm_path = '/environment/packages/spm/12/tpm/head_tpm.nii'

    def brain_mask_pipeline(self, robust=True, f_threshold=0.65,
                            g_threshold=-0.15, **kwargs):
        pipeline = super(UTEStudy, self).brain_mask_pipeline(
            robust=robust, f_threshold=f_threshold,
            g_threshold=g_threshold, **kwargs)
        return pipeline

    def header_info_extraction_pipeline(self, **kwargs):
        return (super(UTEStudy, self).
                header_info_extraction_pipeline_factory(
                    'primary', **kwargs))

    def umap_dcm2nii_conversion_pipeline(self, **kwargs):
        return super(UTEStudy, self).dcm2nii_conversion_pipeline_factory(
            'umap_dcm2nii', 'umap', **kwargs)

    def registration_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Register T1 and T2 to the

        Parameters
        ----------
        """
        pipeline = self.create_pipeline(
            name='registration_pipeline',
            inputs=[DatasetSpec('ute_echo1', dicom_format),
                    DatasetSpec('ute_echo2', dicom_format)],
            outputs=[DatasetSpec('ute1_registered', nifti_format),
                     DatasetSpec('ute2_registered', nifti_gz_format),
                     DatasetSpec('template_to_ute_mat', text_matrix_format),
                     DatasetSpec('ute_to_template_mat', text_matrix_format)],
            description="Register ute images to the template",
            default_options={},
            version=1,
            citations=(fsl_cite),
            options=options)

        echo1_conv = pipeline.create_node(MRConvert(), name='echo1_conv')
        echo1_conv.inputs.out_ext = '.nii.gz'

        pipeline.connect_input('ute_echo1', echo1_conv, 'in_file')

        echo2_conv = pipeline.create_node(MRConvert(), name='echo2_conv')
        echo2_conv.inputs.out_ext = '.nii.gz'

        pipeline.connect_input('ute_echo2', echo2_conv, 'in_file')

        # Create registration node
        registration = pipeline.create_node(
            FLIRT(), name='ute1_registration',
            requirements=[fsl5_req], wall_time=180)

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
        convert_mat = pipeline.create_node(
            ConvertXFM(), name='inverse_matrix_conversion',
            requirements=[fsl5_req], wall_time=10)
        pipeline.connect(
            registration,
            'out_matrix_file',
            convert_mat,
            'in_file')
        convert_mat.inputs.invert_xfm = True

        # UTE_echo_2 transformation
        transform_ute2 = pipeline.create_node(
            ApplyXFM(), name='transform_t2',
            requirements=[fsl5_req], wall_time=10)
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
        pipeline.assert_connected()

        return pipeline

    def segmentation_pipeline(self, **options):  # @UnusedVariable @IgnorePep8

        pipeline = self.create_pipeline(
            name='ute1_segmentation',
            inputs=[DatasetSpec('ute1_registered', nifti_format)],
            outputs=[DatasetSpec('air_mask', nifti_gz_format),
                     DatasetSpec('bones_mask', nifti_gz_format)],
            description="Segmentation of the first echo UTE image",
            default_options={},
            version=1,
            citations=(spm_cite, matlab_cite),
            options=options)

        segmentation = pipeline.create_node(
            NewSegment(), name='ute1_registered_segmentation',
            requirements=[matlab2015_req, spm12_req], wall_time=480)
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

        select_bones_pm = pipeline.create_node(
            Select(), name='select_bones_pm_from_SPM_new_segmentation',
            requirements=[], wall_time=5)
        pipeline.connect(
            segmentation,
            'native_class_images',
            select_bones_pm,
            'inlist')
        select_bones_pm.inputs.index = 3

        select_air_pm = pipeline.create_node(
            Select(), name='select_air_pm_from_SPM_new_segmentation',
            requirements=[], wall_time=5)

        pipeline.connect(
            segmentation,
            'native_class_images',
            select_air_pm,
            'inlist')
        select_air_pm.inputs.index = 5

        threshold_bones = pipeline.create_node(
            Threshold(), name='bones_probabilistic_map_thresholding',
            requirements=[fsl5_req], wall_time=5)
        pipeline.connect(select_bones_pm, 'out', threshold_bones, 'in_file')
        threshold_bones.inputs.output_type = "NIFTI_GZ"
        threshold_bones.inputs.direction = 'below'
        threshold_bones.inputs.thresh = 0.2

        binarize_bones = pipeline.create_node(
            UnaryMaths(), name='bones_probabilistic_map_binarization',
            requirements=[fsl5_req], wall_time=5)
        pipeline.connect(
            threshold_bones,
            'out_file',
            binarize_bones,
            'in_file')
        binarize_bones.inputs.output_type = "NIFTI_GZ"
        binarize_bones.inputs.operation = 'bin'

        threshold_air = pipeline.create_node(
            Threshold(), name='air_probabilistic_maps_thresholding',
            requirements=[fsl5_req], wall_time=5)
        pipeline.connect(select_air_pm, 'out', threshold_air, 'in_file')
        threshold_air.inputs.output_type = "NIFTI_GZ"
        threshold_air.inputs.direction = 'below'
        threshold_air.inputs.thresh = 0.1

        binarize_air = pipeline.create_node(
            UnaryMaths(), name='air_probabilistic_map_binarization',
            requirements=[fsl5_req], wall_time=5)
        pipeline.connect(threshold_air, 'out_file', binarize_air, 'in_file')
        binarize_air.inputs.output_type = "NIFTI_GZ"
        binarize_air.inputs.operation = 'bin'

        pipeline.connect_output('bones_mask', binarize_bones, 'out_file')
        pipeline.connect_output('air_mask', binarize_air, 'out_file')
        pipeline.assert_connected()

        return pipeline

    def umaps_calculation_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='core_umaps_calculation',
            inputs=[DatasetSpec('ute1_registered', nifti_gz_format),
                    DatasetSpec('ute2_registered', nifti_gz_format),
                    DatasetSpec('air_mask', nifti_gz_format),
                    DatasetSpec('bones_mask', nifti_gz_format)],
            outputs=[DatasetSpec('sute_cont_template', nifti_gz_format),
                     DatasetSpec('sute_fix_template', nifti_gz_format)],
            description="Umaps calculation in the template space",
            default_options={},
            version=1,
            citations=(matlab_cite),
            options=options)

        umaps_calculation = pipeline.create_node(
            CoreUmapCalc(), name='umaps_calculation_based_on_masks_and_r2star',
            requirements=[matlab2015_req], wall_time=20)
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
        pipeline.assert_connected()

        return pipeline

    def backwrap_to_ute_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='backwrap_to_ute',
            inputs=[DatasetSpec('ute1_registered', nifti_gz_format),
                    DatasetSpec('ute_echo1', dicom_format),
                    DatasetSpec('umap_ute', dicom_format),
                    DatasetSpec('template_to_ute_mat', text_matrix_format),
                    DatasetSpec('sute_cont_template', nifti_gz_format),
                    DatasetSpec('sute_fix_template', nifti_gz_format)],
            outputs=[DatasetSpec('sute_cont_ute', nifti_gz_format),
                     DatasetSpec('sute_fix_ute', nifti_gz_format)],
            description="Moving umaps back to the UTE space",
            default_options={},
            version=1,
            citations=(matlab_cite),
            options=options)

        echo1_conv = pipeline.create_node(MRConvert(), name='echo1_conv')
        echo1_conv.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('ute_echo1', echo1_conv, 'in_file')

        umap_conv = pipeline.create_node(MRConvert(), name='umap_conv')
        umap_conv.inputs.out_ext = '.nii.gz'
        pipeline.connect_input('umap_ute', umap_conv, 'in_file')

        zero_template_mask = pipeline.create_node(
            BinaryMaths(), name='zero_template_mask',
            requirements=[fsl5_req], wall_time=3)
        pipeline.connect_input(
            'ute1_registered',
            zero_template_mask,
            'in_file')
        zero_template_mask.inputs.operation = "mul"
        zero_template_mask.inputs.operand_value = 0
        zero_template_mask.inputs.output_type = 'NIFTI_GZ'

        region_template_mask = pipeline.create_node(
            FLIRT(), name='region_template_mask',
            requirements=[fsl5_req], wall_time=5)
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

        fill_in_umap = pipeline.create_node(MultiImageMaths(),
                                            name='fill_in_umap',
                                            requirements=[fsl5_req],
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

        sute_fix_ute_space = pipeline.create_node(
            FLIRT(), name='sute_fix_ute_space',
            requirements=[fsl5_req], wall_time=5)
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

        sute_cont_ute_space = pipeline.create_node(
            FLIRT(), name='sute_cont_ute_space',
            requirements=[fsl5_req], wall_time=5)
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

        sute_fix_ute_background = pipeline.create_node(
            MultiImageMaths(), name='sute_fix_ute_background',
            requirements=[fsl5_req], wall_time=5)
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

        sute_cont_ute_background = pipeline.create_node(
            MultiImageMaths(), name='sute_cont_ute_background',
            requirements=[fsl5_req], wall_time=5)
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

        smooth_sute_fix = pipeline.create_node(
            Smooth(), name='smooth_sute_fix',
            requirements=[fsl5_req], wall_time=5)
        smooth_sute_fix.inputs.sigma = 2.
        pipeline.connect(
            sute_fix_ute_background,
            'out_file',
            smooth_sute_fix,
            'in_file')

        smooth_sute_cont = pipeline.create_node(
            Smooth(), name='smooth_sute_cont',
            requirements=[fsl5_req], wall_time=5)
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
        pipeline.assert_connected()

        return pipeline

#     def conversion_to_dicom_pipeline(self, **options):
#
#         pipeline = self.create_pipeline(
#             name='conversion_to_dicom',
#             inputs=[DatasetSpec('sute_cont_ute', nifti_gz_format),
#                     DatasetSpec('sute_fix_ute', nifti_gz_format),
#                     DatasetSpec('umap_ute', dicom_format)],
#             outputs=[DatasetSpec('sute_cont_dicoms', dicom_format),
#                      DatasetSpec('sute_fix_dicoms', dicom_format)],
#             description=(
#                 "Conversing resulted two umaps from nifti to dicom format - "
#                 "parallel implementation"),
#             default_options={},
#             version=1,
#             citations=(),
#             options=options)
#
#         cont_split = pipeline.create_node(Split(), name='cont_split',
#                                           requirements=[fsl5_req])
#         cont_split.inputs.dimension = 'z'
#         fix_split = pipeline.create_node(Split(), name='fix_split',
#                                          requirements=[fsl5_req])
#         fix_split.inputs.dimension = 'z'
#         cont_nii2dicom = pipeline.create_map_node(
#             Nii2Dicom(), name='cont_nii2dicom', iterfield=['in_file',
#                                                            'reference_dicom'],
#             wall_time=20)
#         fix_nii2dicom = pipeline.create_map_node(
#             Nii2Dicom(), name='fix_nii2dicom', iterfield=['in_file',
#                                                           'reference_dicom'],
#             wall_time=20)
#         list_dicoms = pipeline.create_node(ListDir(), name='list_dicoms')
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

    _data_specs = set_specs(
        DatasetSpec('umap', dicom_format),
        DatasetSpec('umap_nifti', nifti_gz_format,
                    umap_dcm2nii_conversion_pipeline),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('ute_echo1', dicom_format),
        DatasetSpec('ute_echo2', dicom_format),
        DatasetSpec('umap_ute', dicom_format),
        DatasetSpec('ute1_registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('ute2_registered', nifti_gz_format, registration_pipeline),
        DatasetSpec('template_to_ute_mat', text_matrix_format,
                    registration_pipeline),
        DatasetSpec('ute_to_template_mat', text_matrix_format,
                    registration_pipeline),
        DatasetSpec('air_mask', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('bones_mask', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('sute_cont_template', nifti_gz_format,
                    umaps_calculation_pipeline),
        DatasetSpec('sute_fix_template', nifti_gz_format,
                    umaps_calculation_pipeline),
        DatasetSpec('sute_fix_ute', nifti_gz_format, backwrap_to_ute_pipeline),
        DatasetSpec('sute_cont_ute', nifti_gz_format,
                    backwrap_to_ute_pipeline),
        inherit_from=MRIStudy.data_specs())
    # The list of study data_specs that are either primary from the scanner
    # (i.e. without a specified pipeline) or generated by processing pipelines
#     _data_specs = set_specs(
#         DatasetSpec(
#             'sute_fix_dicoms',
#             dicom_format,
#             conversion_to_dicom_pipeline),
#         DatasetSpec(
#             'sute_cont_dicoms',
#             dicom_format,
#             conversion_to_dicom_pipeline),
#         inherit_from=MRIStudy.data_specs())


class CoregisteredUTEStudy(MultiStudy):

    __metaclass__ = MultiStudyMetaClass

    ute_basic_preproc_pipeline = MultiStudy.translate(
        'ute', UTEStudy.basic_preproc_pipeline)

    ute_dcm2nii_pipeline = MultiStudy.translate(
        'ute', MRIStudy.dcm2nii_conversion_pipeline)

    ute_umap_dcm2nii_pipeline = MultiStudy.translate(
        'ute', UTEStudy.umap_dcm2nii_conversion_pipeline)

    ute_dcm_info_pipeline = MultiStudy.translate(
        'ute', UTEStudy.header_info_extraction_pipeline,
        override_default_options={'multivol': False})

    ref_bet_pipeline = MultiStudy.translate(
        'reference', MRIStudy.brain_mask_pipeline)

    ref_basic_preproc_pipeline = MultiStudy.translate(
        'reference', MRIStudy.basic_preproc_pipeline,
        override_default_options={'resolution': [1]})

    ute_qform_transform_pipeline = MultiStudy.translate(
        'coreg', CoregisteredStudy.qform_transform_pipeline)

    ute_brain_mask_pipeline = MultiStudy.translate(
        'ute', UTEStudy.brain_mask_pipeline)

    ute_rigid_registration_pipeline = MultiStudy.translate(
        'coreg', CoregisteredStudy.linear_registration_pipeline)

    def motion_mat_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='motion_mat_calculation',
            inputs=[DatasetSpec('ute_reg_mat', text_matrix_format),
                    DatasetSpec('ute_qform_mat', text_matrix_format)],
            outputs=[DatasetSpec('motion_mats', directory_format)],
            description=("utew Motion matrices calculation"),
            default_options={},
            version=1,
            citations=[fsl_cite],
            options=options)

        mm = pipeline.create_node(
            MotionMatCalculation(), name='motion_mats')
        pipeline.connect_input('ute_reg_mat', mm, 'reg_mat')
        pipeline.connect_input('ute_qform_mat', mm, 'qform_mat')
        pipeline.connect_output('motion_mats', mm, 'motion_mats')
        pipeline.assert_connected()
        return pipeline

    def nifti2dcm_conversion_pipeline(self, **options):

        pipeline = self.create_pipeline(
            name='conversion_to_dicom',
            inputs=[DatasetSpec('umap_aligned_niftis', directory_format),
                    DatasetSpec('umap', dicom_format)],
            outputs=[DatasetSpec('umap_aligned_dicoms', directory_format)],
            description=(
                "Conversing aligned umap from nifti to dicom format - "
                "parallel implementation"),
            default_options={},
            version=1,
            citations=(),
            options=options)

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
        pipeline.connect_input('umap_aligned_niftis', list_niftis, 'directory')
        pipeline.connect_input('umap', list_dicoms, 'directory')
        # Connect outputs
        pipeline.connect_output('umap_aligned_dicoms', copy2dir, 'out_dir')

        pipeline.assert_connected()
        return pipeline

    _sub_study_specs = set_specs(
        SubStudySpec('ute', UTEStudy, {
            'ute': 'primary',
            'umap': 'umap',
            'umap_nifti': 'umap_nifti',
            'ute_nifti': 'primary_nifti',
            'ute_preproc': 'preproc',
            'ute_brain': 'masked',
            'ute_brain_mask': 'brain_mask',
            'ped': 'ped',
            'pe_angle': 'pe_angle',
            'tr': 'tr',
            'real_duration': 'real_duration',
            'tot_duration': 'tot_duration',
            'start_time': 'start_time',
            'dcm_info': 'dcm_info'}),
        SubStudySpec('reference', MRIStudy, {
            'reference': 'primary_nifti',
            'ref_preproc': 'preproc',
            'ref_brain': 'masked',
            'ref_brain_mask': 'brain_mask'}),
        SubStudySpec('coreg', CoregisteredStudy, {
            'ute_brain': 'to_register',
            'ref_brain': 'reference',
            'ute_qformed': 'qformed',
            'ute_qform_mat': 'qform_mat',
            'ute_reg': 'registered',
            'ute_reg_mat': 'matrix'}))

    _data_specs = set_specs(
        DatasetSpec('ute', dicom_format),
        DatasetSpec('umap', dicom_format),
        DatasetSpec('umap_nifti', nifti_gz_format, ute_umap_dcm2nii_pipeline),
        DatasetSpec('umap_aligned_niftis', directory_format),
        DatasetSpec('umap_aligned_dicoms', directory_format,
                    nifti2dcm_conversion_pipeline),
        DatasetSpec('ute_nifti', nifti_gz_format, ute_dcm2nii_pipeline),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('ute_preproc', nifti_gz_format,
                    ute_basic_preproc_pipeline),
        DatasetSpec('ute_brain', nifti_gz_format, ute_brain_mask_pipeline),
        DatasetSpec('ute_brain_mask', nifti_gz_format,
                    ute_brain_mask_pipeline),
        DatasetSpec('ref_preproc', nifti_gz_format,
                    ref_basic_preproc_pipeline),
        DatasetSpec('ute_qformed', nifti_gz_format,
                    ute_qform_transform_pipeline),
        DatasetSpec('ute_qform_mat', text_matrix_format,
                    ute_qform_transform_pipeline),
        DatasetSpec('ref_brain', nifti_gz_format, ref_bet_pipeline),
        DatasetSpec('ref_brain_mask', nifti_gz_format,
                    ref_bet_pipeline),
        DatasetSpec('ute_reg', nifti_gz_format,
                    ute_rigid_registration_pipeline),
        DatasetSpec('ute_reg_mat', text_matrix_format,
                    ute_rigid_registration_pipeline),
        DatasetSpec('motion_mats', directory_format,
                    motion_mat_pipeline),
        DatasetSpec('dcm_info', text_format, ute_dcm_info_pipeline),
        FieldSpec('ped', str, ute_dcm_info_pipeline),
        FieldSpec('pe_angle', str, ute_dcm_info_pipeline),
        FieldSpec('tr', float, ute_dcm_info_pipeline),
        FieldSpec('start_time', str, ute_dcm_info_pipeline),
        FieldSpec('real_duration', str, ute_dcm_info_pipeline),
        FieldSpec('tot_duration', str, ute_dcm_info_pipeline))
