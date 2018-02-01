from nipype.pipeline import engine as pe
from nipype.interfaces import fsl
from nianalysis.dataset import DatasetSpec
from nianalysis.study.base import Study, set_dataset_specs
from nianalysis.requirements import Requirement
from nianalysis.citations import fsl_cite, bet_cite, bet2_cite
from nianalysis.data_formats import nifti_gz_format, text_matrix_format
from nianalysis.requirements import fsl5_req
from nipype.interfaces.fsl import FLIRT, FNIRT, Reorient2Std
from nianalysis.utils import get_atlas_path
from nianalysis.exceptions import NiAnalysisError


class MRIStudy(Study):

    def brain_mask_pipeline(self, **options):  # @UnusedVariable
        """
        Generates a whole brain mask using FSL's BET command
        """
        pipeline = self.create_pipeline(
            name='brain_mask',
            inputs=[DatasetSpec('swapped_image', nifti_gz_format)],
            outputs=[DatasetSpec('masked', nifti_gz_format),
                     DatasetSpec('brain_mask', nifti_gz_format)],
            description="Generate brain mask from mr_scan",
            default_options={'robust': False, 'threshold': 0.5,
                             'reduce_bias': False},
            version=1,
            citations=[fsl_cite, bet_cite, bet2_cite],
            options=options)
        # Create mask node
        bet = pipeline.create_node(interface=fsl.BET(), name="bet",
                                   requirements=[fsl5_req])
        bet.inputs.mask = True
        bet.inputs.output_type = 'NIFTI_GZ'
        if pipeline.option('robust'):
            bet.inputs.robust = True
        if pipeline.option('reduce_bias'):
            bet.inputs.reduce_bias = True
        bet.inputs.frac = pipeline.option('threshold')
        # Connect inputs/outputs
        pipeline.connect_input('primary', bet, 'in_file')
        pipeline.connect_output('masked', bet, 'out_file')
        pipeline.connect_output('brain_mask', bet, 'mask_file')
        # Check inputs/outputs are connected
        pipeline.assert_connected()
        return pipeline

    def coregister_to_atlas_pipeline(self, atlas_reg_tool='fnirt',
                                     **options):
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
            inputs=[DatasetSpec('primary', nifti_gz_format),
                    DatasetSpec('brain_mask', nifti_gz_format),
                    DatasetSpec('masked', nifti_gz_format)],
            outputs=[DatasetSpec('coreg_to_atlas', nifti_gz_format),
                     DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format)],
            description=("Nonlinearly registers a MR scan to a standard space,"
                         "e.g. MNI-space"),
            default_options={'atlas': 'MNI152',
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
        pipeline.connect_input('primary', reorient, 'in_file')
        pipeline.connect_input('brain_mask', reorient_mask, 'in_file')
        pipeline.connect_input('masked', reorient_masked, 'in_file')
        # Connect outputs
        pipeline.connect_output('coreg_to_atlas', fnirt, 'warped_file')
        pipeline.connect_output('coreg_to_atlas_coeff', fnirt,
                                'fieldcoeff_file')
        pipeline.assert_connected()
        return pipeline

    def registration_pipeline(self, **options):
        input_datasets = [DatasetSpec('masked', nifti_gz_format),
                          DatasetSpec('reference', nifti_gz_format)]
        output_datasets = [DatasetSpec('reg_file', nifti_gz_format),
                           DatasetSpec('reg_mat', text_matrix_format)]
        reg_type = 'registration'
        return self._registration_factory(input_datasets, output_datasets,
                                          reg_type, **options)

    def applyxfm_pipeline(self, **options):
        input_datasets = [DatasetSpec('masked', nifti_gz_format),
                          DatasetSpec('reference', nifti_gz_format),
                          DatasetSpec('affine_mat', text_matrix_format)]
        output_datasets = [DatasetSpec('applyxfm_reg_file', nifti_gz_format)]
        reg_type = 'applyxfm'
        return self._registration_factory(input_datasets, output_datasets,
                                          reg_type, **options)

    def useqform_pipeline(self, **options):
        input_datasets = [DatasetSpec('masked', nifti_gz_format),
                          DatasetSpec('reference', nifti_gz_format)]
        output_datasets = [DatasetSpec('qform_reg_file', nifti_gz_format),
                           DatasetSpec('qform_mat', text_matrix_format)]
        reg_type = 'useqform'
        return self._registration_factory(input_datasets, output_datasets,
                                          reg_type, **options)

    def _registration_factory(self, input_datasets, output_datasets, reg_type,
                              **options):

        pipeline = self.create_pipeline(
            name='FLIRT_registration',
            inputs=input_datasets,
            outputs=output_datasets,
            description="Registration of the primary image to the reference",
            default_options={'dof': 6, 'cost': 'normmi', 'interp': 'trilinear',
                             'search_cost': 'normmi'},
            version=1,
            citations=[fsl_cite],
            options=options)

        flirt = pipeline.create_node(
            interface=FLIRT(), name="flirt_{}"+reg_type,
            requirements=[fsl5_req])
        flirt.inputs.output_type = 'NIFTI_GZ'
        if pipeline.option('dof'):
            flirt.inputs.dof = pipeline.option('dof')
        if pipeline.option('cost'):
            flirt.inputs.cost = pipeline.option('cost')
        if pipeline.option('interp'):
            flirt.inputs.interp = pipeline.option('interp')
        if pipeline.option('search_cost'):
            flirt.inputs.cost_func = pipeline.option('search_cost')

        pipeline.connect_input('primary', flirt, 'in_file')
        pipeline.connect_input('reference', flirt, 'reference')

        if reg_type == 'useqform':
            flirt.inputs.apply_xfm = True
            flirt.inputs.uses_qform = True
            pipeline.connect_output('qform_reg_file', flirt, 'out_file')
            pipeline.connect_output('qform_mat', flirt, 'out_matrix_file')
        elif reg_type == 'applyxfm':
            flirt.inputs.apply_xfm = True
            pipeline.connect_input('affine_mat', flirt, 'in_matrix_file')
            pipeline.connect_output('applyxfm_reg_file', flirt, 'out_file')
        else:
            pipeline.connect_output('reg_file', flirt, 'out_file')
            pipeline.connect_output('affine_mat', flirt, 'out_matrix_file')

        pipeline.assert_connected()
        return pipeline

    def segmentation_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='FAST_segmentation',
            inputs=[DatasetSpec('ref_brain', nifti_gz_format)],
            outputs=[DatasetSpec('ref_seg', nifti_gz_format)],
            description="White matter segmentation of the reference image",
            default_options={'img_type': 2},
            version=1,
            citations=[fsl_cite],
            options=options)

        fast = pipeline.create_node(fsl.FAST(), name='fast')
        fast.inputs.img_type = pipeline.option('img_type')
        fast.inputs.segments = True
        fast.inputs.out_basename = 'Reference_segmentation'
        pipeline.connect_input('ref_brain', fast, 'in_files')
        pipeline.connect_output('ref_seg', fast, 'tissue_class_files')

        pipeline.assert_connected()
        return pipeline

    def swap_dimensions_pipeline(self, **options):
        pipeline = self.create_pipeline(
            name='fslswapdim_pipeline',
            inputs=[DatasetSpec('primary', nifti_gz_format)],
            outputs=[DatasetSpec('swapped_image', nifti_gz_format)],
            description=("Dimensions swapping to ensure that all the images "
                         "have the same orientations."),
            default_options={'new_dims': ('RL', 'AP', 'IS')},
            version=1,
            citations=[fsl_cite],
            options=options)
        swap = pipeline.create_node(fsl.utils.SwapDimensions(),
                                    name='fslswapdim')
        swap.inputs.new_dims = pipeline.option('new_dims')
        pipeline.connect_input('primary', swap, 'in_file')
        pipeline.connect_output('swapped_image', swap, 'out_file')

        pipeline.assert_connected()
        return pipeline

    _dataset_specs = set_dataset_specs(
        DatasetSpec('primary', nifti_gz_format),
        DatasetSpec('reference', nifti_gz_format),
        DatasetSpec('affine_mat', text_matrix_format),
        DatasetSpec('reg_file', nifti_gz_format, registration_pipeline),
        DatasetSpec('reg_mat', text_matrix_format, registration_pipeline),
        DatasetSpec('qform_reg_file', nifti_gz_format, useqform_pipeline),
        DatasetSpec('qform_mat', text_matrix_format, useqform_pipeline),
        DatasetSpec('applyxfm_reg_file', nifti_gz_format, applyxfm_pipeline),
        DatasetSpec('masked', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('brain_mask', nifti_gz_format, brain_mask_pipeline),
        DatasetSpec('coreg_to_atlas', nifti_gz_format,
                    coregister_to_atlas_pipeline),
        DatasetSpec('coreg_to_atlas_coeff', nifti_gz_format,
                    coregister_to_atlas_pipeline),
        DatasetSpec('ref_seg', nifti_gz_format, segmentation_pipeline),
        DatasetSpec('swapped_image', nifti_gz_format,
                    swap_dimensions_pipeline))
