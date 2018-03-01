from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec, FieldSpec
from nianalysis.study.base import Study, set_data_specs
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import (nifti_gz_format, dicom_format,
                                     text_format, directory_format, gif_format)
from nianalysis.requirements import fsl5_req, mrtrix3_req, fsl509_req
from nipype.interfaces.fsl import (FLIRT, FNIRT, Reorient2Std)
from nianalysis.utils import get_atlas_path
from nianalysis.exceptions import NiAnalysisError
from mbianalysis.interfaces.mrtrix.transform import MRResize
from mbianalysis.interfaces.custom.dicom import (DicomHeaderInfoExtraction)
from nipype.interfaces.utility import Split
from nianalysis.interfaces.mrtrix import MRConvert
from mbianalysis.interfaces.fsl import FSLSlices


class MRIStudy(Study):

    def brain_mask_pipeline(self, **options):
        bet_method = options.get('bet_method', 'fsl_bet')
        if bet_method == 'fsl_bet':
            pipeline = self._fsl_bet_brain_mask_pipeline(**options)
        elif bet_method == 'optibet':
            pipeline = self._optiBET_brain_mask_pipeline(**options)
        else:
            raise NiAnalysisError("Unrecognised brain extraction tool '{}'"
                                  .format(bet_method))
        return pipeline

    def _fsl_bet_brain_mask_pipeline(self, **options):
        """
        Generates a whole brain mask using FSL's BET command.
        """
        pipeline = self.create_pipeline(
            name='brain_mask',
            inputs=[DatasetSpec('preproc', nifti_gz_format)],
            outputs=[DatasetSpec('masked', nifti_gz_format),
                     DatasetSpec('brain_mask', nifti_gz_format)],
            description="Generate brain mask from mr_scan",
            default_options={'robust': True, 'f_threshold': 0.5,
                             'reduce_bias': False, 'g_threshold': 0.0,
                             'bet_method': 'fsl_bet'},
            version=1,
            citations=[fsl_cite, bet_cite, bet2_cite],
            options=options)
        # Create mask node
        bet = pipeline.create_node(interface=fsl.BET(), name="bet",
                                   requirements=[fsl509_req])
        bet.inputs.mask = True
        bet.inputs.output_type = 'NIFTI_GZ'
        if pipeline.option('robust'):
            bet.inputs.robust = True
        if pipeline.option('reduce_bias'):
            bet.inputs.reduce_bias = True
        bet.inputs.frac = pipeline.option('f_threshold')
        bet.inputs.vertical_gradient = pipeline.option('g_threshold')
        # Connect inputs/outputs
        pipeline.connect_input('preproc', bet, 'in_file')
        pipeline.connect_output('masked', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        pipeline.assert_connected()
        return pipeline

    def _optiBET_brain_mask_pipeline(self, **options):
        """
        Generates a whole brain mask using optiBET.
        """
        gen_report_default = False
        outputs = [DatasetSpec('masked', nifti_gz_format),
                   DatasetSpec('brain_mask', nifti_gz_format)]
        if options.get('gen_report', gen_report_default):
            outputs.append(DatasetSpec('optiBET_report', gif_format))
        pipeline = self.create_pipeline(
            name='brain_mask',
            inputs=[DatasetSpec('preproc', nifti_gz_format)],
            outputs=outputs,
            description="Generate brain mask from mr_scan",
            default_options={'robust': True, 'f_threshold': 0.5,
                             'reduce_bias': False, 'g_threshold': 0.0,
                             'gen_report': gen_report_default,
                             'bet_method': 'optibet'},
            version=1,
            citations=[fsl_cite, bet_cite, bet2_cite],
            options=options)
        bet1 = pipeline.create_node(
            fsl.BET(frac=0.1, reduce_bias=True), name='bet', wall_time=15,
            requirements=[fsl5_req])
        pipeline.connect_input('preproc', bet1, 'in_file')
        flirt = pipeline.create_node(
            FLIRT(out_matrix_file='linear_mat.mat',
                  out_file='linear_reg.nii.gz', searchr_x=[-30, 30],
                  searchr_y=[-30, 30], searchr_z=[-30, 30]), name='flirt',
            wall_time=5, requirements=[fsl5_req])
        flirt.inputs.reference = pipeline.option('MNI_template')
        pipeline.connect(bet1, 'out_file', flirt, 'in_file')
        fnirt = pipeline.create_node(
            FNIRT(config_file='T1_2_MNI152_2mm',
                  fieldcoeff_file='warp_file.nii.gz'), name='fnirt',
            wall_time=15, requirements=[fsl5_req])
        fnirt.inputs.ref_file = pipeline.option('MNI_template')
        pipeline.connect(flirt, 'out_matrix_file', fnirt, 'affine_file')
        pipeline.connect_input('preproc', fnirt, 'in_file')
        invwarp = pipeline.create_node(
            fsl.InvWarp(), name='invwarp', wall_time=10,
            requirements=[fsl5_req])
        pipeline.connect(fnirt, 'fieldcoeff_file', invwarp, 'warp')
        pipeline.connect_input('preproc', invwarp, 'reference')
        applywarp = pipeline.create_node(
            fsl.ApplyWarp(interp='nn', out_file='warped_file.nii.gz'),
            name='applywarp', wall_time=5, requirements=[fsl5_req])
        applywarp.inputs.in_file = pipeline.option('MNI_template_mask')
        pipeline.connect_input('preproc', applywarp, 'ref_file')
        pipeline.connect(invwarp, 'inverse_warp', applywarp, 'field_file')
        maths1 = pipeline.create_node(
            fsl.ImageMaths(suffix='_optiBET_brain_mask', op_string='-bin'),
            name='binarize', wall_time=5, requirements=[fsl5_req])
        pipeline.connect(applywarp, 'out_file', maths1, 'in_file')
        maths2 = pipeline.create_node(
            fsl.ImageMaths(suffix='_optiBET_brain', op_string='-mas'),
            name='mask', wall_time=5, requirements=[fsl5_req])
        pipeline.connect_input('preproc', maths2, 'in_file')
        pipeline.connect(maths1, 'out_file', maths2, 'in_file2')
        if pipeline.option('gen_report'):
            slices = pipeline.create_node(
                FSLSlices(), name='slices', wall_time=5,
                requirements=[fsl5_req])
            slices.inputs.outname = 'optiBET_report'
            pipeline.connect_input('preproc', slices, 'im1')
            pipeline.connect(maths2, 'out_file', slices, 'im2')
            pipeline.connect_output('optiBET_report', slices, 'report')

        pipeline.connect_output('brain_mask', maths1, 'out_file')
        pipeline.connect_output('masked', maths2, 'out_file')

        pipeline.assert_connected()
        return pipeline

    def coregister_to_atlas_pipeline(self, **options):
        atlas_reg_tool = options.get('atlas_reg_tool', 'fnirt')
        if atlas_reg_tool == 'fnirt':
            pipeline = self._fsl_fnirt_to_atlas_pipeline(**options)
        else:
            raise NiAnalysisError("Unrecognised coregistration tool '{}'"
                                  .format(atlas_reg_tool))
        return pipeline

    def _fsl_fnirt_to_atlas_pipeline(self, **options):  # @UnusedVariable @IgnorePep8
        """
        Registers a MR scan to a refernce MR scan using FSL's nonlinear FNIRT
        command

        Parameters
        ----------
        atlas : Which atlas to use, can be one of 'mni_nl6'
        """
        pipeline = self.create_pipeline(
            name='coregister_to_atlas_fnirt',
            inputs=[DatasetSpec('preproc', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('masked', nifti_gz_format)],
            outputs=[DatasetSpec('coreg_to_atlas', nifti_gz_format),
                     DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format)],
            description=("Nonlinearly registers a MR scan to a standard space,"
                         "e.g. MNI-space"),
            default_options={'atlas_reg_tool': 'fnirt',
                             'atlas': 'MNI152',
                             'resolution': '2mm',
                             'intensity_model': 'global_non_linear_with_bias',
                             'subsampling': [4, 4, 2, 2, 1, 1]},
            version=1,
            citations=[fsl_cite],
            options=options)
        # Get the reference atlas from FSL directory
        ref_atlas = get_atlas_path(pipeline.option('atlas'), 'image',
                                   resolution=pipeline.option('resolution'))
        ref_mask = get_atlas_path(pipeline.option('atlas'), 'mask_dilated',
                                  resolution=pipeline.option('resolution'))
        ref_masked = get_atlas_path(pipeline.option('atlas'), 'masked',
                                    resolution=pipeline.option('resolution'))
        # Basic reorientation to standard MNI space
        reorient = pipeline.create_node(Reorient2Std(), name='reorient',
                                        requirements=[fsl5_req])
        reorient.inputs.output_type = 'NIFTI_GZ'
        reorient_mask = pipeline.create_node(
            Reorient2Std(), name='reorient_mask', requirements=[fsl5_req])
        reorient_mask.inputs.output_type = 'NIFTI_GZ'
        reorient_masked = pipeline.create_node(
            Reorient2Std(), name='reorient_masked', requirements=[fsl5_req])
        reorient_masked.inputs.output_type = 'NIFTI_GZ'
        # Affine transformation to MNI space
        flirt = pipeline.create_node(interface=FLIRT(), name='flirt',
                                     requirements=[fsl5_req],
                                     wall_time=5)
        flirt.inputs.reference = ref_masked
        flirt.inputs.dof = 12
        flirt.inputs.output_type = 'NIFTI_GZ'
        # Nonlinear transformation to MNI space
        fnirt = pipeline.create_node(interface=FNIRT(), name='fnirt',
                                     requirements=[fsl5_req],
                                     wall_time=60)
        fnirt.inputs.ref_file = ref_atlas
        fnirt.inputs.refmask_file = ref_mask
        fnirt.inputs.output_type = 'NIFTI_GZ'
        intensity_model = pipeline.option('intensity_model')
        if intensity_model is None:
            intensity_model = 'none'
        fnirt.inputs.intensity_mapping_model = intensity_model
        fnirt.inputs.subsampling_scheme = pipeline.option('subsampling')
        fnirt.inputs.fieldcoeff_file = True
        fnirt.inputs.in_fwhm = [8, 6, 5, 4.5, 3, 2]
        fnirt.inputs.ref_fwhm = [8, 6, 5, 4, 2, 0]
        fnirt.inputs.regularization_lambda = [300, 150, 100, 50, 40, 30]
        fnirt.inputs.apply_intensity_mapping = [1, 1, 1, 1, 1, 0]
        fnirt.inputs.max_nonlin_iter = [5, 5, 5, 5, 5, 10]
        # Apply mask if corresponding subsampling scheme is 1
        # (i.e. 1-to-1 resolution) otherwise don't.
        apply_mask = [int(s == 1) for s in pipeline.option('subsampling')]
        fnirt.inputs.apply_inmask = apply_mask
        fnirt.inputs.apply_refmask = apply_mask
        # Connect nodes
        pipeline.connect(reorient_masked, 'out_file', flirt, 'in_file')
        pipeline.connect(reorient, 'out_file', fnirt, 'in_file')
        pipeline.connect(reorient_mask, 'out_file', fnirt, 'inmask_file')
        pipeline.connect(flirt, 'out_matrix_file', fnirt, 'affine_file')
        # Set registration options
        # TODO: Need to work out which options to use
        # Connect inputs
        pipeline.connect_input('preproc', reorient, 'in_file')
        pipeline.connect_input('brain_mask', reorient_mask, 'in_file')
        pipeline.connect_input('masked', reorient_masked, 'in_file')
        # Connect outputs
        pipeline.connect_output('coreg_to_atlas', fnirt, 'warped_file')
        pipeline.connect_output('coreg_to_atlas_coeff', fnirt,
                                'fieldcoeff_file')
        pipeline.assert_connected()
        return pipeline

    def segmentation_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='FAST_segmentation',
            inputs=[DatasetSpec('masked', nifti_gz_format)],
            outputs=[DatasetSpec('wm_seg', nifti_gz_format)],
            description="White matter segmentation of the reference image",
            default_options={'img_type': 2},
            version=1,
            citations=[fsl_cite],
            options=options)

        fast = pipeline.create_node(fsl.FAST(), name='fast',
                                    requirements=[fsl509_req])
        fast.inputs.img_type = pipeline.option('img_type')
        fast.inputs.segments = True
        fast.inputs.out_basename = 'Reference_segmentation'
        pipeline.connect_input('masked', fast, 'in_files')
        split = pipeline.create_node(Split(), name='split')
        split.inputs.splits = [1, 1, 1]
        split.inputs.squeeze = True
        pipeline.connect(fast, 'tissue_class_files', split, 'inlist')
        pipeline.connect_output('wm_seg', split, 'out3')

        pipeline.assert_connected()
        return pipeline

    def basic_preproc_pipeline(self, **options):
        """
        Performs basic preprocessing, such as swapping dimensions into
        standard orientation and resampling (if required)

        Options
        -------
        new_dims : tuple(str)[3]
            A 3-tuple with the new orientation of the image (see FSL
            swap dim)
        resolution : list(float)[3] | None
            New resolution of the image. If None no resampling is
            performed
        """
        pipeline = self.create_pipeline(
            name='fslswapdim_pipeline',
            inputs=[DatasetSpec('primary_nifti', nifti_gz_format)],
            outputs=[DatasetSpec('preproc', nifti_gz_format)],
            description=("Dimensions swapping to ensure that all the images "
                         "have the same orientations."),
            default_options={'new_dims': ('RL', 'AP', 'IS'),
                             'resolution': None},
            version=1,
            citations=[fsl_cite],
            options=options)
        swap = pipeline.create_node(fsl.utils.SwapDimensions(),
                                    name='fslswapdim',
                                    requirements=[fsl509_req])
        swap.inputs.new_dims = pipeline.option('new_dims')
        pipeline.connect_input('primary_nifti', swap, 'in_file')
        if pipeline.option('resolution') is not None:
            resample = pipeline.create_node(MRResize(), name="resample",
                                            requirements=[mrtrix3_req])
            resample.inputs.voxel = pipeline.option('resolution')
            pipeline.connect(swap, 'out_file', resample, 'in_file')
            pipeline.connect_output('preproc', resample, 'out_file')
        else:
            pipeline.connect_output('preproc', swap, 'out_file')

        pipeline.assert_connected()
        return pipeline

    def header_info_extraction_pipeline(self, **options):
        return self.header_info_extraction_pipeline_factory('dicom_file',
                                                            **options)

    def header_info_extraction_pipeline_factory(self, dcm_in_name, ref=False,
                                                **options):
        output_files = [FieldSpec('tr', dtype=float),
                        FieldSpec('start_time', dtype=str),
                        FieldSpec('tot_duration', dtype=str),
                        FieldSpec('real_duration', dtype=str),
                        FieldSpec('ped', dtype=str),
                        FieldSpec('pe_angle', dtype=str),
                        DatasetSpec('dcm_info', text_format)]
        if ref:
            output_files.append(DatasetSpec('ref_motion_mats',
                                            directory_format))

        pipeline = self.create_pipeline(
            name='header_info_extraction',
            inputs=[DatasetSpec(dcm_in_name, dicom_format)],
            outputs=output_files,
            description=("Pipeline to extract the most important scan "
                         "information from the image header"),
            default_options={'multivol': True},
            version=1,
            citations=[],
            options=options)
        hd_extraction = pipeline.create_node(DicomHeaderInfoExtraction(),
                                             name='hd_info_extraction')
        hd_extraction.inputs.multivol = pipeline.option('multivol')
        hd_extraction.inputs.reference = ref
        pipeline.connect_input(dcm_in_name, hd_extraction, 'dicom_folder')
        pipeline.connect_output('tr', hd_extraction, 'tr')
        pipeline.connect_output('start_time', hd_extraction, 'start_time')
        pipeline.connect_output(
            'tot_duration', hd_extraction, 'tot_duration')
        pipeline.connect_output(
            'real_duration', hd_extraction, 'real_duration')
        pipeline.connect_output('ped', hd_extraction, 'ped')
        pipeline.connect_output('pe_angle', hd_extraction, 'pe_angle')
        pipeline.connect_output('dcm_info', hd_extraction, 'dcm_info')
        if ref:
            pipeline.connect_output('ref_motion_mats', hd_extraction,
                                    'ref_motion_mats')
        pipeline.assert_connected()
        return pipeline

    def dcm2nii_conversion_pipeline(self, **kwargs):
        return self.dcm2nii_conversion_pipeline_factory(
                    'dcm2nii_conversion', 'primary', **kwargs)

    def dcm2nii_conversion_pipeline_factory(self, name, dcm_in_name,
                                            **options):
        pipeline = self.create_pipeline(
            name=name,
            inputs=[DatasetSpec(dcm_in_name, dicom_format)],
            outputs=[DatasetSpec(dcm_in_name+'_nifti', nifti_gz_format)],
            description=("DICOM to NIFTI conversion."),
            default_options={},
            version=1,
            citations=[],
            options=options)

        converter = pipeline.create_node(MRConvert(), name='converter1',
                                         requirements=[mrtrix3_req])
        converter.inputs.out_ext = '.nii.gz'
        pipeline.connect_input(dcm_in_name, converter, 'in_file')
        pipeline.connect_output(
            dcm_in_name+'_nifti', converter, 'out_file')

        pipeline.assert_connected()
        return pipeline

    _data_specs = set_data_specs(
        DatasetSpec('primary', dicom_format),
        DatasetSpec('primary_nifti', nifti_gz_format),
        DatasetSpec('dicom_dwi', dicom_format),
        DatasetSpec('dicom_dwi_1', dicom_format),
        DatasetSpec('preproc', nifti_gz_format,
                    basic_preproc_pipeline),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('brain_mask', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('coreg_to_atlas', nifti_gz_format,
                    coregister_to_atlas_pipeline),
        DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    coregister_to_atlas_pipeline),
        DatasetSpec('wm_seg', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('dicom_file', dicom_format),
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
        DatasetSpec('dcm_info', text_format, header_info_extraction_pipeline)
        )


class MotionReferenceStudy(MRIStudy):
    def header_info_extraction_pipeline(self, reference=True, multivol=False,
                                        **kwargs):
        return (super(MotionReferenceStudy, self).
                header_info_extraction_pipeline_factory(
                    'primary', ref=reference, multivol=multivol,
                    **kwargs))
    _data_specs = set_data_specs(
        DatasetSpec('ref_motion_mats', directory_format,
                    header_info_extraction_pipeline),
        inherit_from=MRIStudy.data_specs())
